# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Interleaved RB analysis class.
"""
from typing import List, Union

import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import AnalysisResultData, FitVal
from .rb_analysis import RBAnalysis


class LeakageRBAnalysis(RBAnalysis):
    r"""A class to analyze interleaved randomized benchmarking experiment.

    # section: overview
        This analysis takes only two series for standard and interleaved RB curve fitting.
        From the fit :math:`\alpha` and :math:`\alpha_c` value this analysis estimates
        the error per Clifford (EPC) of the interleaved gate.

        The EPC estimate is obtained using the equation

        .. math::

            r_{\mathcal{C}}^{\text{est}} =
                \frac{\left(d-1\right)\left(1-\alpha_{\overline{\mathcal{C}}}/\alpha\right)}{d}

        The systematic error bounds are given by

        .. math::

            E = \min\left\{
                \begin{array}{c}
                    \frac{\left(d-1\right)\left[\left|\alpha-\alpha_{\overline{\mathcal{C}}}\right|
                    +\left(1-\alpha\right)\right]}{d} \\
                    \frac{2\left(d^{2}-1\right)\left(1-\alpha\right)}
                    {\alpha d^{2}}+\frac{4\sqrt{1-\alpha}\sqrt{d^{2}-1}}{\alpha}
                \end{array}
            \right.

        See Ref. [1] for more details.

    # section: fit_model
        The fit is based on the following decay functions:

        Fit model for standard RB

        .. math::

            F(x) = a \alpha^{x} + b

        Fit model for interleaved RB

        .. math::

            F(x) = a (\alpha_c \alpha)^{x_2} + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by the average :math:`a` of the standard and interleaved RB.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by the average :math:`b` of the standard and interleaved RB.
                Usually equivalent to :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by the slope of :math:`(y - b)^{-x}` of the first and the
                second data point of the standard RB.
            bounds: [0, 1]
        defpar \alpha_c:
            desc: Ratio of the depolarizing parameter of interleaved RB to standard RB curve.
            init_guess: Estimate :math:`\alpha' = \alpha_c \alpha` from the
                interleaved RB curve, then divide this by the initial guess of :math:`\alpha`.
            bounds: [0, 1]

    # section: reference
        .. ref_arxiv:: 1 1203.4550

    """

    __series__ = [
        curve.SeriesDef(
            name="Normal",
            fit_func=lambda x, lambda_1, lambda_2, A_1, B_1, C_1, A_2, B_2: 
            curve.fit_function.bi_exponential_decay(
                x, amp_1=B_1, lamb_1=-1.0, base_1=lambda_1,
                amp_2=C_1, lamb_2=-1.0, base_2=lambda_2, baseline=A_1
            ),
            filter_kwargs={"interleaved": False},
            plot_color="red",
            plot_symbol=".",
            plot_fit_uncertainty=True,
        ),
        
        curve.SeriesDef(
            name="Inverted",
            fit_func=lambda x, lambda_1, lambda_2, A_1, B_1, C_1, A_2, B_2: 
            curve.fit_function.bi_exponential_decay(
                x, amp_1=B_2, lamb_1=-1.0, base_1=lambda_1, 
                amp_2=C_1*(-1.0), lamb_2=-1.0, base_2=lambda_2, baseline=A_2
            ),
            filter_kwargs={"interleaved": True},
            plot_color="orange",
            plot_symbol="^",
            plot_fit_uncertainty=True,
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        default_options = super()._default_options()
        default_options.xlabel = "Clifford Length"
        default_options.ylabel = "P(0)"
        default_options.result_parameters = ["A_1", "A_2", "lambda_1", "lambda_2"]
        default_options.error_dict = None
        default_options.epg_1_qubit = None
        default_options.gate_error_ratio = None
        return default_options

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(
            lambda_1=(0, 1),
            lambda_2=(0, 1),
            A_1=(0, 1),
            B_1=(0, 1),
            C_1=(0, 1),
            A_2=(0, 1),
            B_2=(0, 1),
        )
            
        user_opt.p0.set_if_empty(
            lambda_1=0.99,
            lambda_2=0.8,
            A_1=0.5,
            B_1=0.5,
            C_1=0.01,
            A_2=0.01,
            B_2=0.01,
        )   

        return user_opt
        
    def _extra_database_entry(self, fit_data: curve.FitData) -> List[AnalysisResultData]:
        extra_entries = []
        
        nrb = 2 ** self._num_qubits
        print(fit_data)
        A = fit_data.fitval("A_1").value+fit_data.fitval("A_2").value
        lambda_1 = fit_data.fitval("lambda_1").value
        lambda_2 = fit_data.fitval("lambda_2").value
        
        # Calculate leakage and sweepage
        leakage = (1 - A) * (1 - lambda_1)
        seepage = A * (1 - lambda_1)
        
        # Calculate EPC
        epc = 1 - 1 / nrb * ((nrb-1) * lambda_2 + 1 - leakage)

        extra_entries.append(
            AnalysisResultData(
                name="Leakage_err",
                value=leakage,
                chisq=fit_data.reduced_chisq,
                quality=self._evaluate_quality(fit_data),
            )
        )
        
        extra_entries.append(
            AnalysisResultData(
                name="Seepage_err",
                value=seepage,
                chisq=fit_data.reduced_chisq,
                quality=self._evaluate_quality(fit_data),
            )
        )        

        extra_entries.append(
            AnalysisResultData(
                name="EPC",
                value=epc,
                chisq=fit_data.reduced_chisq,
                quality=self._evaluate_quality(fit_data),
            )
        )
        
        return extra_entries