"""Automated verification suite for VisionLab CV and Bike Sharing Regression ML pipeline."""

import os
import unittest

import numpy as np
import pandas as pd

from modules.bike_sharing_ml import (
    execute_five_fold_cv,
    get_candidate_models,
    load_bike_dataset,
    prepare_data_split,
    save_and_generate_plots,
)
from modules.face_detection import run_vision_pipeline
from modules.pdf_generator import build_a4_pdf_report
from utils.sample_generator import generate_sample_portrait


class TestMLSuiteVariant(unittest.TestCase):
    """Full end-to-end unit test suite for Zameel's ML Suite."""

    def test_visionlab_detection_pipeline(self) -> None:
        """Verifies synthetic portrait generation, face detection, and privacy blur."""
        img = generate_sample_portrait("single")
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)

        annotated, detections, latency_ms = run_vision_pipeline(
            rgb_image=img,
            scale_factor=1.12,
            min_neighbors=3,
            min_size=30,
            detect_eyes=True,
            detect_smiles=True,
            privacy_blur=False,
            box_color_hex="#10B981",
        )
        self.assertEqual(annotated.shape, img.shape)
        self.assertGreaterEqual(len(detections), 1)
        self.assertGreater(latency_ms, 0.0)

        # Test privacy blur
        blurred, blur_detections, _ = run_vision_pipeline(
            rgb_image=img,
            privacy_blur=True,
        )
        self.assertEqual(len(blur_detections), len(detections))
        self.assertFalse(np.array_equal(annotated, blurred))

    def test_bike_sharing_dataset_loading(self) -> None:
        """Verifies dataset shape, non-empty features, and no missing values."""
        df = load_bike_dataset()
        self.assertGreater(len(df), 17000)
        self.assertIn("cnt", df.columns)
        self.assertIn("temp", df.columns)
        self.assertIn("hr", df.columns)
        self.assertEqual(df["cnt"].isna().sum(), 0)

    def test_bike_sharing_pipeline_and_cv(self) -> None:
        """Verifies train-test split, 5-fold CV execution, and plot generation."""
        df = load_bike_dataset()
        split = prepare_data_split(df, test_size=0.20, random_state=42)

        self.assertEqual(len(split["y_train"]) + len(split["y_test"]), len(df))
        self.assertEqual(split["X_train_scaled"].shape[1], 12)

        # Benchmark candidate models
        models = get_candidate_models()
        self.assertIn("Random Forest Regressor", models)

        bench_df, artifacts, best_name = execute_five_fold_cv(
            models=models,
            X_train_scaled=split["X_train_scaled"],
            y_train=split["y_train"],
            X_test_scaled=split["X_test_scaled"],
            y_test=split["y_test"],
        )

        self.assertEqual(len(bench_df), len(models))
        self.assertTrue(best_name in models)

        best_row = bench_df.iloc[0]
        self.assertGreater(best_row["CV R2 Mean"], 0.85)
        self.assertGreater(best_row["Test R2"], 0.85)

        # Generate plots
        best_model = artifacts[best_name]["model"]
        test_preds = artifacts[best_name]["test_preds"]
        plots = save_and_generate_plots(
            df=df,
            bench_df=bench_df,
            best_model_name=best_name,
            best_model=best_model,
            test_preds=test_preds,
            y_test=split["y_test"],
            test_df=split["test_df"],
        )

        self.assertEqual(len(plots), 6)
        for key, p_path in plots.items():
            self.assertTrue(os.path.exists(p_path), f"Plot {key} was not generated at {p_path}")

        # Generate A4 PDF Report
        best_metrics = {
            "cv_r2_mean": float(best_row["CV R2 Mean"]),
            "test_r2": float(best_row["Test R2"]),
            "test_rmse": float(best_row["Test RMSE"]),
        }
        pdf_path = "evaluation_plots/Test_Bike_Sharing_Report.pdf"
        pdf_bytes = build_a4_pdf_report(
            benchmark_df=bench_df,
            best_model_name=best_name,
            best_metrics=best_metrics,
            feature_importance_df=None,
            plot_paths=plots,
            author_name="Zameel",
            output_path=pdf_path,
        )

        self.assertGreater(len(pdf_bytes), 10000)
        self.assertTrue(os.path.exists(pdf_path))


if __name__ == "__main__":
    unittest.main()
