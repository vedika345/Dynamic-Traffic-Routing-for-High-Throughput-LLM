"""
Router
=======

Implements the routing *policy*: given a classified prompt
(task_type + complexity score), decide which "model role"
(see config/router_config.yaml -> models:) should handle it,
then delegate to the ModelPool to actually run it.

This is where the "cost vs quality" tradeoff lives:
  - cheap/simple prompts -> small, fast local model
  - hard reasoning / code / creative prompts -> bigger, specialized model
"""


class Router:
    def __init__(self, classifier, model_pool, complexity_threshold: float = 0.6):
        self.classifier = classifier
        self.model_pool = model_pool
        self.complexity_threshold = complexity_threshold

    def select_model_key(self, task_type: str, complexity: float) -> str:
        """
        Pure decision function: (task_type, complexity) -> model role.
        Kept separate from `route()` so it's easy to unit test.
        """
        if task_type == "code":
            return "code"

        if task_type == "math_reasoning":
            return "complex_reasoning" if complexity >= self.complexity_threshold else "general"

        if task_type == "creative":
            return "creative"

        if task_type == "summarization":
            return "general"

        # task_type == "general_chat" (default)
        return "simple_chat" if complexity < self.complexity_threshold else "general"

    def route(self, prompt: str) -> dict:
        task_type, complexity = self.classifier.classify(prompt)
        model_key = self.select_model_key(task_type, complexity)

        result = self.model_pool.generate(model_key, prompt)
        result.update({
            "task_type": task_type,
            "complexity": complexity,
            "model_key": model_key,
        })
        return result
