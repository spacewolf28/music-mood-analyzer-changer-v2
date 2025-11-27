# -*- coding: utf-8 -*-

from backend.inference.analyze import Analyzer


def evaluate_generated(original_path, generated_path):
    print("\n====== Evaluate Generated Music ======")

    analyzer = Analyzer()

    orig = analyzer.extract(original_path)
    gen = analyzer.extract(generated_path)

    print("\nOriginal style:", orig["style"])
    print("Generated style:", gen["style"])

    print("\nOriginal emotion:", orig["emotion"])
    print("Generated emotion:", gen["emotion"])

    print("=====================================\n")

    return {"orig": orig, "gen": gen}
