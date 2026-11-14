
def _get_submodule(module_name, submodule_name):
    """Get the submodule docstring and automatically add the hook.

    `module_name` is e.g. `sklearn.feature_extraction`, and `submodule_name` is e.g.
    `image`, so we get the docstring and hook for `sklearn.feature_extraction.image`
    submodule. `module_name` is used to reset the current module because autosummary
    automatically changes the current module.
    """
    lines = [
        f".. automodule:: {module_name}.{submodule_name}",
        f".. currentmodule:: {module_name}",
    ]
    return "\n\n".join(lines)

API_REFERENCE = {
    "mlquantify": {
        "short_summary": "A library for quantifying machine learning models.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "get_config",
                    "set_config",
                    "config_context",
                ],
            }
        ],
    },
    "mlquantify.base": {
        "short_summary": "Base classes functions for quantifiers.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "BaseQuantifier",
                    "MetaquantifierMixin",
                    "ProtocolMixin",
                ],
            }
        ],
    },
    "mlquantify.base_aggregative": {
        "short_summary": "Aggregative quantifiers base classes.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "AggregationMixin",
                    "SoftPredictionMixin",
                    "CrispPredictionMixin"
                ],
            }
        ],
    },
    "mlquantify.datasets": {
        "short_summary": "Synthetic generators and real-world dataset loaders.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "make_quantification",
                    "fetch_mushroom", "fetch_banknote_authentication", "fetch_haberman_survival", "fetch_miniboone",
                    "fetch_digits_optical_penbased", "fetch_dry_bean", "fetch_covertype", "fetch_yeast",
                    "fetch_sensorless_drive", "fetch_statlog_shuttle", "fetch_wine_quality", "fetch_online_news_popularity",
                    "fetch_pima_diabetes", "fetch_electricity_elec2", "fetch_airlines",
                    "fetch_newsgroups20", "fetch_imdb", "fetch_multidomain_sentiment", "fetch_sentiment140", "fetch_rcv1_v2",
                    "fetch_mnist_usps", "fetch_cifar10",
                    "fetch_planetoid_cora_citeseer_pubmed",
                    "fetch_sea_concepts",
                    "fetch_lequa2024",
                    "Bunch", "get_data_home", "fetch_remote",
                ],
            }
        ],
    },
    "mlquantify.multiclass": {
        "short_summary": "Multiclass definitions and utilities.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "binary_quantifier",
                    "BinaryQuantifier",
                    "MulticlassStrategy",
                    "register_strategy",
                    "get_strategy",
                    "available_strategies",
                ],
            }
        ],
    },
    "mlquantify.confidence": {
        "short_summary": "Confidence Regions for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Confidence Regions",
                "autosummary": [
                    "BaseConfidenceRegion",
                    "ConfidenceInterval",
                    "ConfidenceEllipseSimplex",
                    "ConfidenceEllipseCLR",
                    "construct_confidence_region"
                ],
            }
        ],
    },
    "mlquantify.visualization": {
        "short_summary": "Plotting utilities for quantification results.",
        "description": None,
        "sections": [
            {
                "title": "Multiple-sample displays",
                "autosummary": [
                    "DiagonalDisplay",
                    "BiasDisplay",
                    "ErrorByShiftDisplay",
                ],
            },
            {
                "title": "Single-sample displays",
                "autosummary": [
                    "PrevalenceDisplay",
                    "ConfidenceRegionDisplay",
                ],
            },
        ],
    },
    "mlquantify.counting": {
        "short_summary": "Counting methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Counting Methods",
                "autosummary": [
                    "CC",
                    "PCC",
                    "ACC",
                    "ThresholdAdjustment",
                    "TAC",
                    "TX",
                    "TMAX",
                    "T50",
                    "MS",
                    "MS2",
                    "FM",
                    "GACC",
                    "GPACC",
                    "evaluate_thresholds",
                    "compute_tpr",
                    "compute_fpr",
                    "compute_table",
                ],
            }
        ],
    },
    "mlquantify.likelihood": {
        "short_summary": "Likelihood methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Likelihood Methods",
                "autosummary": [
                    "CDE",
                    "EMQ",
                    "MLPE",
                ],
            }
        ],
    },
    "mlquantify.matching": {
        "short_summary": "Distribution matching methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Matching Methods",
                "autosummary": [
                    "BaseMatchingQuantifier",
                    "MatchingHistogramQuantifier",
                    "DyS",
                    "HDy",
                    "HDx",
                    "SORD",
                    "MatchingKernelQuantifier",
                    "MMD_RKHS",
                    "KDEyQuantifier",
                    "KDEyML",
                    "KDEyHD",
                    "KDEyCS",
                    "GKDEyML",
                    "GHDx",
                    "GHDy",
                    "SMM",
                    "EDy",
                    "EDx",
                ],
            }
        ],
    },
    "mlquantify.neighbors": {
        "short_summary": "Neighbor-based methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Neighbor-based Methods",
                "autosummary": [
                    "PWK",
                    "PWKAC",
                ],
            }
        ],
    },
    "mlquantify.compose": {
        "short_summary": "Composable quantification methods.",
        "description": None,
        "sections": [
            {
                "title": "Composable Methods",
                "autosummary": [
                    "BaseComposeQuantifier",
                    "LinearComposeQuantifier",
                    "LikelihoodComposeQuantifier",
                    "ComposeQuantifier",
                ],
            }
        ],
    },
    "mlquantify.losses": {
        "short_summary": "Loss functions used by quantifiers.",
        "description": None,
        "sections": [
            {
                "title": "Loss Functions",
                "autosummary": [
                    "BaseLoss",
                    "DistanceLoss",
                    "LeastSquaresLoss",
                    "HellingerSurrogateLoss",
                    "EnergyLoss",
                    "NegativeLogLikelihoodLoss",
                    "MixtureNegativeLogLikelihoodLoss",
                    "RegularizedMixtureNLLLoss",
                    "normalize_distribution",
                    "get_loss",
                ],
            }
        ],
    },
    "mlquantify.representations": {
        "short_summary": "Representation strategies for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Representations",
                "autosummary": [
                    "BaseRepresentation",
                    "HistogramRepresentation",
                    "KDERepresentation",
                    "DistanceRepresentation",
                    "KernelMeanRepresentation",
                    "PredictionRepresentation",
                    "HardPredictionRepresentation",
                    "SoftPredictionRepresentation",
                ],
            }
        ],
    },
    "mlquantify.solvers": {
        "short_summary": "Optimization helpers for prevalence estimation.",
        "description": None,
        "sections": [
            {
                "title": "Solvers",
                "autosummary": [
                    "solve_binary",
                    "ternary_search",
                    "solve_simplex",
                    "minimize_prevalence",
                    "minimize_prevalence_blocks",
                ],
            }
        ],
    },
    "mlquantify.calibration": {
        "short_summary": "Post-hoc calibration of classifier posteriors.",
        "description": (
            "Scaling-based calibrators that rescale a classifier's logits to "
            "minimise the held-out negative log-likelihood (temperature and "
            "vector scaling), improving probabilistic quantifiers such as EMQ."
        ),
        "sections": [
            {
                "title": "Calibration",
                "autosummary": [
                    "Calibrator",
                    "ClassifierCalibrator",
                    "QuantifierCalibrator",
                ],
            }
        ],
    },
    "mlquantify.neural": {
        "short_summary": "Neural quantification methods.",
        "description": None,
        "sections": [
            {
                "title": "Neural Methods",
                "autosummary": [
                    "QuaNet",
                ],
            }
        ],
    },
    "mlquantify.meta": {
        "short_summary": "Meta methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Meta Methods",
                "autosummary": [
                    "EnsembleQ", 
                    "QuaDapt", 
                    "AggregativeBootstrap"
                ],
            }
        ],
    },
    "mlquantify.metrics": {
        "short_summary": "Metrics for quantification.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "AE",
                    "SE",
                    "MAE",
                    "MSE",
                    "KLD",
                    "RAE",
                    "NAE",
                    "NRAE",
                    "NKLD",
                    "NMD",
                    "RNOD",
                    "VSE",
                    "CvM_L1",
                ],
            }
        ],
    },
    "mlquantify.model_selection": {
        "short_summary": "Model selection methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "GridSearchQ",
                    "BaseProtocol",
                    "APP",
                    "NPP",
                    "UPP",
                    "PPP",
                    "apply_protocol",
                ],
            }
        ],
    },
    "mlquantify.utils": {
        "short_summary": "Utility functions for quantification.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "get_prev_from_labels",
                    "normalize_prevalence",
                    "load_quantifier",
                    "make_prevs",
                    "apply_cross_validation",
                    "simplex_uniform_kraemer",
                    "simplex_grid_sampling",
                    "simplex_uniform_sampling",
                    "get_indexes_with_prevalence"
                ],
            }
        ]
    },
}
