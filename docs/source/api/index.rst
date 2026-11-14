:html_theme.sidebar_secondary.remove:

.. _api_ref:

=============
API Reference
=============

This is the class and function reference of mlquantify. Please refer to the
:ref:`full user guide <user_guide>` for further details, as the raw specifications of
classes and functions may not be enough to give full guidelines on their use. For
reference on core concepts, see the :ref:`Foundations <quantification_foundations>` guide.

.. toctree::
  :maxdepth: 2
  :hidden:


  mlquantify
  mlquantify.base
  mlquantify.base_aggregative
  mlquantify.calibration
  mlquantify.compose
  mlquantify.confidence
  mlquantify.counting
  mlquantify.datasets
  mlquantify.likelihood
  mlquantify.losses
  mlquantify.matching
  mlquantify.meta
  mlquantify.metrics
  mlquantify.model_selection
  mlquantify.multiclass
  mlquantify.neighbors
  mlquantify.neural
  mlquantify.representations
  mlquantify.solvers
  mlquantify.utils
  mlquantify.visualization

.. list-table::
  :header-rows: 1
  :class: apisearch-table

  * - Object
    - Description








  * - :obj:`~mlquantify.get_config`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify

        .. autoshortsummary:: mlquantify.get_config

        .. div:: caption

          :mod:`mlquantify`





  * - :obj:`~mlquantify.set_config`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify

        .. autoshortsummary:: mlquantify.set_config

        .. div:: caption

          :mod:`mlquantify`





  * - :obj:`~mlquantify.config_context`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify

        .. autoshortsummary:: mlquantify.config_context

        .. div:: caption

          :mod:`mlquantify`









  * - :obj:`~mlquantify.base.BaseQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.base

        .. autoshortsummary:: mlquantify.base.BaseQuantifier

        .. div:: caption

          :mod:`mlquantify.base`





  * - :obj:`~mlquantify.base.MetaquantifierMixin`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.base

        .. autoshortsummary:: mlquantify.base.MetaquantifierMixin

        .. div:: caption

          :mod:`mlquantify.base`





  * - :obj:`~mlquantify.base.ProtocolMixin`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.base

        .. autoshortsummary:: mlquantify.base.ProtocolMixin

        .. div:: caption

          :mod:`mlquantify.base`









  * - :obj:`~mlquantify.base_aggregative.AggregationMixin`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.base_aggregative

        .. autoshortsummary:: mlquantify.base_aggregative.AggregationMixin

        .. div:: caption

          :mod:`mlquantify.base_aggregative`





  * - :obj:`~mlquantify.base_aggregative.SoftPredictionMixin`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.base_aggregative

        .. autoshortsummary:: mlquantify.base_aggregative.SoftPredictionMixin

        .. div:: caption

          :mod:`mlquantify.base_aggregative`





  * - :obj:`~mlquantify.base_aggregative.CrispPredictionMixin`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.base_aggregative

        .. autoshortsummary:: mlquantify.base_aggregative.CrispPredictionMixin

        .. div:: caption

          :mod:`mlquantify.base_aggregative`









  * - :obj:`~mlquantify.calibration.Calibrator`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.calibration

        .. autoshortsummary:: mlquantify.calibration.Calibrator

        .. div:: caption

          :mod:`mlquantify.calibration`





  * - :obj:`~mlquantify.calibration.ClassifierCalibrator`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.calibration

        .. autoshortsummary:: mlquantify.calibration.ClassifierCalibrator

        .. div:: caption

          :mod:`mlquantify.calibration`





  * - :obj:`~mlquantify.calibration.QuantifierCalibrator`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.calibration

        .. autoshortsummary:: mlquantify.calibration.QuantifierCalibrator

        .. div:: caption

          :mod:`mlquantify.calibration`









  * - :obj:`~mlquantify.compose.BaseComposeQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.compose

        .. autoshortsummary:: mlquantify.compose.BaseComposeQuantifier

        .. div:: caption

          :mod:`mlquantify.compose`





  * - :obj:`~mlquantify.compose.LinearComposeQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.compose

        .. autoshortsummary:: mlquantify.compose.LinearComposeQuantifier

        .. div:: caption

          :mod:`mlquantify.compose`





  * - :obj:`~mlquantify.compose.LikelihoodComposeQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.compose

        .. autoshortsummary:: mlquantify.compose.LikelihoodComposeQuantifier

        .. div:: caption

          :mod:`mlquantify.compose`





  * - :obj:`~mlquantify.compose.ComposeQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.compose

        .. autoshortsummary:: mlquantify.compose.ComposeQuantifier

        .. div:: caption

          :mod:`mlquantify.compose`









  * - :obj:`~mlquantify.confidence.BaseConfidenceRegion`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.confidence

        .. autoshortsummary:: mlquantify.confidence.BaseConfidenceRegion

        .. div:: caption

          :mod:`mlquantify.confidence`





  * - :obj:`~mlquantify.confidence.ConfidenceInterval`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.confidence

        .. autoshortsummary:: mlquantify.confidence.ConfidenceInterval

        .. div:: caption

          :mod:`mlquantify.confidence`





  * - :obj:`~mlquantify.confidence.ConfidenceEllipseSimplex`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.confidence

        .. autoshortsummary:: mlquantify.confidence.ConfidenceEllipseSimplex

        .. div:: caption

          :mod:`mlquantify.confidence`





  * - :obj:`~mlquantify.confidence.ConfidenceEllipseCLR`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.confidence

        .. autoshortsummary:: mlquantify.confidence.ConfidenceEllipseCLR

        .. div:: caption

          :mod:`mlquantify.confidence`





  * - :obj:`~mlquantify.confidence.construct_confidence_region`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.confidence

        .. autoshortsummary:: mlquantify.confidence.construct_confidence_region

        .. div:: caption

          :mod:`mlquantify.confidence`









  * - :obj:`~mlquantify.counting.CC`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.CC

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.PCC`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.PCC

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.ACC`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.ACC

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.ThresholdAdjustment`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.ThresholdAdjustment

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.TAC`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.TAC

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.TX`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.TX

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.TMAX`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.TMAX

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.T50`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.T50

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.MS`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.MS

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.MS2`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.MS2

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.FM`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.FM

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.GACC`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.GACC

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.GPACC`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.GPACC

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.evaluate_thresholds`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.evaluate_thresholds

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.compute_tpr`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.compute_tpr

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.compute_fpr`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.compute_fpr

        .. div:: caption

          :mod:`mlquantify.counting`





  * - :obj:`~mlquantify.counting.compute_table`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.counting

        .. autoshortsummary:: mlquantify.counting.compute_table

        .. div:: caption

          :mod:`mlquantify.counting`









  * - :obj:`~mlquantify.datasets.make_quantification`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.make_quantification

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_mushroom`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_mushroom

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_banknote_authentication`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_banknote_authentication

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_haberman_survival`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_haberman_survival

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_miniboone`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_miniboone

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_digits_optical_penbased`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_digits_optical_penbased

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_dry_bean`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_dry_bean

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_covertype`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_covertype

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_yeast`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_yeast

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_sensorless_drive`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_sensorless_drive

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_statlog_shuttle`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_statlog_shuttle

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_wine_quality`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_wine_quality

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_online_news_popularity`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_online_news_popularity

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_pima_diabetes`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_pima_diabetes

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_electricity_elec2`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_electricity_elec2

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_airlines`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_airlines

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_newsgroups20`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_newsgroups20

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_imdb`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_imdb

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_multidomain_sentiment`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_multidomain_sentiment

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_sentiment140`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_sentiment140

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_rcv1_v2`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_rcv1_v2

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_mnist_usps`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_mnist_usps

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_cifar10`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_cifar10

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_planetoid_cora_citeseer_pubmed`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_planetoid_cora_citeseer_pubmed

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_sea_concepts`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_sea_concepts

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_lequa2024`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_lequa2024

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.Bunch`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.Bunch

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.get_data_home`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.get_data_home

        .. div:: caption

          :mod:`mlquantify.datasets`





  * - :obj:`~mlquantify.datasets.fetch_remote`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.datasets

        .. autoshortsummary:: mlquantify.datasets.fetch_remote

        .. div:: caption

          :mod:`mlquantify.datasets`









  * - :obj:`~mlquantify.likelihood.CDE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.likelihood

        .. autoshortsummary:: mlquantify.likelihood.CDE

        .. div:: caption

          :mod:`mlquantify.likelihood`





  * - :obj:`~mlquantify.likelihood.EMQ`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.likelihood

        .. autoshortsummary:: mlquantify.likelihood.EMQ

        .. div:: caption

          :mod:`mlquantify.likelihood`





  * - :obj:`~mlquantify.likelihood.MLPE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.likelihood

        .. autoshortsummary:: mlquantify.likelihood.MLPE

        .. div:: caption

          :mod:`mlquantify.likelihood`









  * - :obj:`~mlquantify.losses.BaseLoss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.BaseLoss

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.DistanceLoss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.DistanceLoss

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.LeastSquaresLoss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.LeastSquaresLoss

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.HellingerSurrogateLoss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.HellingerSurrogateLoss

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.EnergyLoss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.EnergyLoss

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.NegativeLogLikelihoodLoss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.NegativeLogLikelihoodLoss

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.MixtureNegativeLogLikelihoodLoss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.MixtureNegativeLogLikelihoodLoss

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.RegularizedMixtureNLLLoss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.RegularizedMixtureNLLLoss

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.normalize_distribution`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.normalize_distribution

        .. div:: caption

          :mod:`mlquantify.losses`





  * - :obj:`~mlquantify.losses.get_loss`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.losses

        .. autoshortsummary:: mlquantify.losses.get_loss

        .. div:: caption

          :mod:`mlquantify.losses`









  * - :obj:`~mlquantify.matching.BaseMatchingQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.BaseMatchingQuantifier

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.MatchingHistogramQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.MatchingHistogramQuantifier

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.DyS`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.DyS

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.HDy`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.HDy

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.HDx`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.HDx

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.SORD`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.SORD

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.MatchingKernelQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.MatchingKernelQuantifier

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.MMD_RKHS`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.MMD_RKHS

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.KDEyQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.KDEyQuantifier

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.KDEyML`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.KDEyML

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.KDEyHD`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.KDEyHD

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.KDEyCS`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.KDEyCS

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.GKDEyML`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.GKDEyML

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.GHDx`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.GHDx

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.GHDy`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.GHDy

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.SMM`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.SMM

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.EDy`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.EDy

        .. div:: caption

          :mod:`mlquantify.matching`





  * - :obj:`~mlquantify.matching.EDx`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.matching

        .. autoshortsummary:: mlquantify.matching.EDx

        .. div:: caption

          :mod:`mlquantify.matching`









  * - :obj:`~mlquantify.meta.EnsembleQ`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.meta

        .. autoshortsummary:: mlquantify.meta.EnsembleQ

        .. div:: caption

          :mod:`mlquantify.meta`





  * - :obj:`~mlquantify.meta.QuaDapt`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.meta

        .. autoshortsummary:: mlquantify.meta.QuaDapt

        .. div:: caption

          :mod:`mlquantify.meta`





  * - :obj:`~mlquantify.meta.AggregativeBootstrap`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.meta

        .. autoshortsummary:: mlquantify.meta.AggregativeBootstrap

        .. div:: caption

          :mod:`mlquantify.meta`









  * - :obj:`~mlquantify.metrics.AE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.AE

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.SE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.SE

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.MAE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.MAE

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.MSE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.MSE

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.KLD`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.KLD

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.RAE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.RAE

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.NAE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.NAE

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.NRAE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.NRAE

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.NKLD`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.NKLD

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.NMD`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.NMD

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.RNOD`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.RNOD

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.VSE`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.VSE

        .. div:: caption

          :mod:`mlquantify.metrics`





  * - :obj:`~mlquantify.metrics.CvM_L1`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.metrics

        .. autoshortsummary:: mlquantify.metrics.CvM_L1

        .. div:: caption

          :mod:`mlquantify.metrics`









  * - :obj:`~mlquantify.model_selection.GridSearchQ`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.model_selection

        .. autoshortsummary:: mlquantify.model_selection.GridSearchQ

        .. div:: caption

          :mod:`mlquantify.model_selection`





  * - :obj:`~mlquantify.model_selection.BaseProtocol`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.model_selection

        .. autoshortsummary:: mlquantify.model_selection.BaseProtocol

        .. div:: caption

          :mod:`mlquantify.model_selection`





  * - :obj:`~mlquantify.model_selection.APP`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.model_selection

        .. autoshortsummary:: mlquantify.model_selection.APP

        .. div:: caption

          :mod:`mlquantify.model_selection`





  * - :obj:`~mlquantify.model_selection.NPP`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.model_selection

        .. autoshortsummary:: mlquantify.model_selection.NPP

        .. div:: caption

          :mod:`mlquantify.model_selection`





  * - :obj:`~mlquantify.model_selection.UPP`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.model_selection

        .. autoshortsummary:: mlquantify.model_selection.UPP

        .. div:: caption

          :mod:`mlquantify.model_selection`





  * - :obj:`~mlquantify.model_selection.PPP`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.model_selection

        .. autoshortsummary:: mlquantify.model_selection.PPP

        .. div:: caption

          :mod:`mlquantify.model_selection`





  * - :obj:`~mlquantify.model_selection.apply_protocol`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.model_selection

        .. autoshortsummary:: mlquantify.model_selection.apply_protocol

        .. div:: caption

          :mod:`mlquantify.model_selection`









  * - :obj:`~mlquantify.multiclass.binary_quantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.multiclass

        .. autoshortsummary:: mlquantify.multiclass.binary_quantifier

        .. div:: caption

          :mod:`mlquantify.multiclass`





  * - :obj:`~mlquantify.multiclass.BinaryQuantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.multiclass

        .. autoshortsummary:: mlquantify.multiclass.BinaryQuantifier

        .. div:: caption

          :mod:`mlquantify.multiclass`





  * - :obj:`~mlquantify.multiclass.MulticlassStrategy`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.multiclass

        .. autoshortsummary:: mlquantify.multiclass.MulticlassStrategy

        .. div:: caption

          :mod:`mlquantify.multiclass`





  * - :obj:`~mlquantify.multiclass.register_strategy`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.multiclass

        .. autoshortsummary:: mlquantify.multiclass.register_strategy

        .. div:: caption

          :mod:`mlquantify.multiclass`





  * - :obj:`~mlquantify.multiclass.get_strategy`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.multiclass

        .. autoshortsummary:: mlquantify.multiclass.get_strategy

        .. div:: caption

          :mod:`mlquantify.multiclass`





  * - :obj:`~mlquantify.multiclass.available_strategies`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.multiclass

        .. autoshortsummary:: mlquantify.multiclass.available_strategies

        .. div:: caption

          :mod:`mlquantify.multiclass`









  * - :obj:`~mlquantify.neighbors.PWK`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.neighbors

        .. autoshortsummary:: mlquantify.neighbors.PWK

        .. div:: caption

          :mod:`mlquantify.neighbors`








  * - :obj:`~mlquantify.neighbors.PWKAC`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.neighbors

        .. autoshortsummary:: mlquantify.neighbors.PWKAC

        .. div:: caption

          :mod:`mlquantify.neighbors`









  * - :obj:`~mlquantify.neural.QuaNet`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.neural

        .. autoshortsummary:: mlquantify.neural.QuaNet

        .. div:: caption

          :mod:`mlquantify.neural`









  * - :obj:`~mlquantify.representations.BaseRepresentation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.representations

        .. autoshortsummary:: mlquantify.representations.BaseRepresentation

        .. div:: caption

          :mod:`mlquantify.representations`





  * - :obj:`~mlquantify.representations.HistogramRepresentation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.representations

        .. autoshortsummary:: mlquantify.representations.HistogramRepresentation

        .. div:: caption

          :mod:`mlquantify.representations`





  * - :obj:`~mlquantify.representations.KDERepresentation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.representations

        .. autoshortsummary:: mlquantify.representations.KDERepresentation

        .. div:: caption

          :mod:`mlquantify.representations`





  * - :obj:`~mlquantify.representations.DistanceRepresentation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.representations

        .. autoshortsummary:: mlquantify.representations.DistanceRepresentation

        .. div:: caption

          :mod:`mlquantify.representations`





  * - :obj:`~mlquantify.representations.KernelMeanRepresentation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.representations

        .. autoshortsummary:: mlquantify.representations.KernelMeanRepresentation

        .. div:: caption

          :mod:`mlquantify.representations`





  * - :obj:`~mlquantify.representations.PredictionRepresentation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.representations

        .. autoshortsummary:: mlquantify.representations.PredictionRepresentation

        .. div:: caption

          :mod:`mlquantify.representations`





  * - :obj:`~mlquantify.representations.HardPredictionRepresentation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.representations

        .. autoshortsummary:: mlquantify.representations.HardPredictionRepresentation

        .. div:: caption

          :mod:`mlquantify.representations`





  * - :obj:`~mlquantify.representations.SoftPredictionRepresentation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.representations

        .. autoshortsummary:: mlquantify.representations.SoftPredictionRepresentation

        .. div:: caption

          :mod:`mlquantify.representations`









  * - :obj:`~mlquantify.solvers.solve_binary`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.solvers

        .. autoshortsummary:: mlquantify.solvers.solve_binary

        .. div:: caption

          :mod:`mlquantify.solvers`





  * - :obj:`~mlquantify.solvers.ternary_search`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.solvers

        .. autoshortsummary:: mlquantify.solvers.ternary_search

        .. div:: caption

          :mod:`mlquantify.solvers`





  * - :obj:`~mlquantify.solvers.solve_simplex`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.solvers

        .. autoshortsummary:: mlquantify.solvers.solve_simplex

        .. div:: caption

          :mod:`mlquantify.solvers`





  * - :obj:`~mlquantify.solvers.minimize_prevalence`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.solvers

        .. autoshortsummary:: mlquantify.solvers.minimize_prevalence

        .. div:: caption

          :mod:`mlquantify.solvers`





  * - :obj:`~mlquantify.solvers.minimize_prevalence_blocks`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.solvers

        .. autoshortsummary:: mlquantify.solvers.minimize_prevalence_blocks

        .. div:: caption

          :mod:`mlquantify.solvers`









  * - :obj:`~mlquantify.utils.get_prev_from_labels`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.get_prev_from_labels

        .. div:: caption

          :mod:`mlquantify.utils`





  * - :obj:`~mlquantify.utils.normalize_prevalence`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.normalize_prevalence

        .. div:: caption

          :mod:`mlquantify.utils`





  * - :obj:`~mlquantify.utils.load_quantifier`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.load_quantifier

        .. div:: caption

          :mod:`mlquantify.utils`





  * - :obj:`~mlquantify.utils.make_prevs`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.make_prevs

        .. div:: caption

          :mod:`mlquantify.utils`





  * - :obj:`~mlquantify.utils.apply_cross_validation`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.apply_cross_validation

        .. div:: caption

          :mod:`mlquantify.utils`





  * - :obj:`~mlquantify.utils.simplex_uniform_kraemer`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.simplex_uniform_kraemer

        .. div:: caption

          :mod:`mlquantify.utils`





  * - :obj:`~mlquantify.utils.simplex_grid_sampling`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.simplex_grid_sampling

        .. div:: caption

          :mod:`mlquantify.utils`





  * - :obj:`~mlquantify.utils.simplex_uniform_sampling`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.simplex_uniform_sampling

        .. div:: caption

          :mod:`mlquantify.utils`





  * - :obj:`~mlquantify.utils.get_indexes_with_prevalence`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.utils

        .. autoshortsummary:: mlquantify.utils.get_indexes_with_prevalence

        .. div:: caption

          :mod:`mlquantify.utils`









  * - :obj:`~mlquantify.visualization.DiagonalDisplay`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.visualization

        .. autoshortsummary:: mlquantify.visualization.DiagonalDisplay

        .. div:: caption

          :mod:`mlquantify.visualization`





  * - :obj:`~mlquantify.visualization.BiasDisplay`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.visualization

        .. autoshortsummary:: mlquantify.visualization.BiasDisplay

        .. div:: caption

          :mod:`mlquantify.visualization`





  * - :obj:`~mlquantify.visualization.ErrorByShiftDisplay`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.visualization

        .. autoshortsummary:: mlquantify.visualization.ErrorByShiftDisplay

        .. div:: caption

          :mod:`mlquantify.visualization`







  * - :obj:`~mlquantify.visualization.PrevalenceDisplay`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.visualization

        .. autoshortsummary:: mlquantify.visualization.PrevalenceDisplay

        .. div:: caption

          :mod:`mlquantify.visualization`





  * - :obj:`~mlquantify.visualization.ConfidenceRegionDisplay`

    - .. div:: sk-apisearch-desc

        .. currentmodule:: mlquantify.visualization

        .. autoshortsummary:: mlquantify.visualization.ConfidenceRegionDisplay

        .. div:: caption

          :mod:`mlquantify.visualization`




