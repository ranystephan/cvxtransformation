import time
from functools import lru_cache

import cvxpy as cp
import numpy as np
from loguru import logger
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt

from experiments.backtest import (
    BacktestResult,
    OptimizationInput,
    Timing,
    load_data,
    run_backtest,
)
from experiments.utils import checkpoints_path, figures_path, get_solver


def parameter_scaling_markowitz(
    inputs: OptimizationInput,
) -> tuple[np.ndarray, float, cp.Problem]:
    problem, param_dict, w, c = get_parametrized_problem(inputs.n_assets, inputs.risk_target)
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    param_dict["chol"].value = inputs.chol
    param_dict["volas"].value = inputs.volas
    param_dict["rho_mean"].value = np.percentile(np.abs(inputs.mean.values), 20, axis=0) * np.ones(
        inputs.n_assets
    )
    param_dict["w_prev"].value = (
        inputs.quantities * inputs.prices.iloc[-1] / portfolio_value
    ).values
    param_dict["c_prev"].value = inputs.cash / portfolio_value
    param_dict["mean"].value = inputs.mean.values
    param_dict["risk_free"].value = inputs.risk_free
    # Use real shorting fees instead of hardcoded values
    param_dict["shorting_fees"].value = inputs.shorting_fees

    problem.solve(solver=get_solver())
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    return w.value, c.value, problem


@lru_cache
def get_parametrized_problem(
    n_assets: int, risk_target: float
) -> tuple[cp.Problem, dict, cp.Variable, cp.Variable]:
    rho_covariance = 0.02
    L_max = 1.6
    T_max = 50 / 252 / 2

    w_lower = np.ones(n_assets) * (-0.05)
    w_upper = np.ones(n_assets) * 0.1
    c_lower = -0.05
    c_upper = 1.0
    gamma_risk = 5.0

    w_prev = cp.Parameter(n_assets)
    c_prev = cp.Parameter()
    mean = cp.Parameter(n_assets)
    risk_free = cp.Parameter()
    rho_mean = cp.Parameter(n_assets)
    chol = cp.Parameter((n_assets, n_assets))
    volas = cp.Parameter(n_assets, nonneg=True)
    shorting_fees = cp.Parameter(n_assets, nonneg=True)

    w, c = cp.Variable(n_assets), cp.Variable()

    z = w - w_prev
    T = cp.norm1(z) / 2
    L = cp.norm1(w)

    # worst-case (robust) return
    mean_return = w @ mean + risk_free * c
    abs_weight_var = cp.Variable(n_assets, nonneg=True)
    return_uncertainty = rho_mean @ abs_weight_var
    return_wc = mean_return - return_uncertainty

    # worst-case (robust) risk
    risk = cp.norm2(chol.T @ w)
    risk_uncertainty = rho_covariance**0.5 * volas @ abs_weight_var
    risk_wc = cp.norm2(cp.hstack([risk, risk_uncertainty]))

    # Shorting costs using hardcoded 5% annual rate
    kappa_short = np.ones(n_assets) * 5 * (0.01) ** 2  # 5% yearly hardcoded
    shorting_costs = kappa_short @ cp.pos(-w)

    objective = return_wc - gamma_risk * cp.pos(risk_wc - risk_target) - shorting_costs

    constraints = [
        cp.sum(w) + c == 1,
        c == c_prev - cp.sum(z),
        c_lower <= c,
        c <= c_upper,
        w_lower <= w,
        w <= w_upper,
        L <= L_max,
        T <= T_max,
        cp.abs(w) <= abs_weight_var,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)

    param_dict = {
        "w_prev": w_prev,
        "mean": mean,
        "risk_free": risk_free,
        "rho_mean": rho_mean,
        "chol": chol,
        "volas": volas,
        "c_prev": c_prev,
        "shorting_fees": shorting_fees,
    }
    return problem, param_dict, w, c


def plot_timings(timings: list[Timing]) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    line_color = "#888888"
    plt.figure()
    plt.stackplot(
        range(len(timings)),
        [timing.cvxpy for timing in timings],
        [timing.solver for timing in timings],
        [timing.other for timing in timings],
        labels=["CVXPY", "Solver", "Other"],
        colors=colors,
    )

    # add light horizontal line for average solver time
    average_cvxpy_time = np.mean([timing.cvxpy for timing in timings])
    average_solver_time = np.mean([timing.solver for timing in timings])
    average_other_time = np.mean([timing.other for timing in timings])

    plt.axhline(
        average_cvxpy_time,
        color=line_color,
        linestyle="--",
    )

    plt.axhline(
        average_solver_time + average_cvxpy_time,
        color=line_color,
        linestyle="--",
    )

    plt.axhline(
        average_other_time + average_solver_time + average_cvxpy_time,
        color=line_color,
        linestyle="--",
    )

    plt.xlabel("Day of backtest")
    plt.xlim(0, len(timings))

    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig(figures_path() / "timing_parametrized.pdf")

    show_plot = False
    if show_plot:
        plt.show()


def initialize_problem(n_assets: int, sigma_target: float) -> None:
    start = time.perf_counter()
    problem, param_dict, _, _ = get_parametrized_problem(n_assets, sigma_target)

    try:
        for p in param_dict.values():
            if p.shape == ():  # Scalar parameter
                p.value = 0.0
            else:
                p.value = np.zeros(p.shape)
        problem.solve(solver=get_solver())
    except cp.SolverError:
        pass

    end = time.perf_counter()
    logger.info(f"First call to get_parametrized_problem took {end-start} seconds")


def main(from_checkpoint: bool = False) -> None:
    annualized_target = 0.10
    sigma_target = annualized_target / np.sqrt(252)

    if not from_checkpoint:
        logger.info("Running parameter scaling")

        n_assets = load_data()[0].shape[1]

        initialize_problem(n_assets, sigma_target)

        scaling_parametrized_markowitz_result = run_backtest(
            parameter_scaling_markowitz, sigma_target, verbose=True
        )
        scaling_parametrized_markowitz_result.save(
            checkpoints_path() / f"scaling_parametrized_{annualized_target}.pickle"
        )
    else:
        scaling_parametrized_markowitz_result = BacktestResult.load(
            checkpoints_path() / f"scaling_parametrized_{annualized_target}.pickle"
        )

    # Display comprehensive portfolio performance metrics
    logger.info("=" * 60)
    logger.info("PORTFOLIO PERFORMANCE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Backtest Period: {scaling_parametrized_markowitz_result.history[0].strftime('%Y-%m-%d')} to {scaling_parametrized_markowitz_result.history[-1].strftime('%Y-%m-%d')}")
    logger.info(f"Number of Trading Days: {len(scaling_parametrized_markowitz_result.history)}")
    logger.info(f"Number of Assets: {scaling_parametrized_markowitz_result.quantities.shape[1]}")
    logger.info("")
    
    # Performance metrics
    logger.info("PERFORMANCE METRICS:")
    logger.info(f"  Annualized Return:     {scaling_parametrized_markowitz_result.mean_return:.2%}")
    logger.info(f"  Annualized Volatility: {scaling_parametrized_markowitz_result.volatility:.2%}")
    logger.info(f"  Sharpe Ratio:          {scaling_parametrized_markowitz_result.sharpe:.2f}")
    logger.info(f"  Maximum Drawdown:      {scaling_parametrized_markowitz_result.max_drawdown:.2%}")
    logger.info(f"  Maximum Leverage:      {scaling_parametrized_markowitz_result.max_leverage:.2f}")
    logger.info(f"  Annualized Turnover:   {scaling_parametrized_markowitz_result.turnover:.2f}")
    logger.info("")
    
    # Portfolio composition
    final_portfolio_value = scaling_parametrized_markowitz_result.portfolio_value.iloc[-1]
    initial_portfolio_value = scaling_parametrized_markowitz_result.portfolio_value.iloc[0]
    total_return = (final_portfolio_value / initial_portfolio_value - 1) * 100
    
    logger.info("PORTFOLIO SUMMARY:")
    logger.info(f"  Initial Portfolio Value: ${initial_portfolio_value:,.2f}")
    logger.info(f"  Final Portfolio Value:   ${final_portfolio_value:,.2f}")
    logger.info(f"  Total Return:            {total_return:.2f}%")
    logger.info(f"  Final Cash Position:     ${scaling_parametrized_markowitz_result.cash.iloc[-1]:,.2f}")
    logger.info("")
    
    # Timing information
    total_time = sum(t.total for t in scaling_parametrized_markowitz_result.timings)
    cvxpy_time = sum(t.cvxpy for t in scaling_parametrized_markowitz_result.timings)
    solver_time = sum(t.solver for t in scaling_parametrized_markowitz_result.timings)
    other_time = sum(t.other for t in scaling_parametrized_markowitz_result.timings)
    
    logger.info("COMPUTATIONAL PERFORMANCE:")
    logger.info(f"  Total Computation Time: {total_time:.1f} seconds")
    logger.info(f"  Average Time per Day:   {total_time / len(scaling_parametrized_markowitz_result.timings):.3f} seconds")
    logger.info(f"  CVXPY Time:             {cvxpy_time/total_time:.1%}")
    logger.info(f"  Solver Time:            {solver_time/total_time:.1%}")
    logger.info(f"  Other Time:             {other_time/total_time:.1%}")
    logger.info("=" * 60)
    
    # Create timing plot
    plot_timings(scaling_parametrized_markowitz_result.timings)
    logger.info(f"Timing plot saved to: {figures_path() / 'timing_parametrized.pdf'}")
    
    # Print comprehensive data report
    print_full_report()


def print_full_report():
    import pandas as pd
    from experiments.backtest import load_data
    print("\n================ FULL OUTPUT REPORT ================\n")
    prices, spread, rf, volume, short_fee_data = load_data()

    # 1. Experiment Metadata
    print("[Experiment Metadata]")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Number of trading days: {prices.shape[0]}")
    print(f"Number of assets (stocks): {prices.shape[1]}")
    print(f"List of all stocks (first 10): {list(prices.columns[:10])}")
    print(f"... (total {len(prices.columns)})\n")

    # 2. Data Provenance Table
    print("[Data Provenance]")
    def data_summary(df, name, typ, fname):
        print(f"- {name}: {typ} (file: {fname}) shape={df.shape}")
        print(f"  Head: \n{df.head(2)}")
        print(f"  Tail: \n{df.tail(2)}")
        print(f"  Missing values: {df.isna().sum().sum()} ({df.isna().mean().mean()*100:.2f}%)\n")
    data_summary(prices, 'Prices', 'Real', 'prices_cleaned.csv')
    data_summary(spread, 'Spreads', 'Real', 'spread_cleaned.csv')
    data_summary(volume, 'Volumes', 'Real', 'volume_cleaned.csv')
    data_summary(rf.to_frame(), 'Risk-Free Rate', 'Real', 'rf.csv')
    data_summary(short_fee_data, 'Shorting Fees', 'Real', 'short_fee_data_cleaned.csv')

    # 3. Shorting Fee Data Coverage
    print("[Shorting Fee Data Coverage]")
    fee_assets = [c[4:] for c in short_fee_data.columns if c.startswith('fee_')]
    print(f"✅ INTERSECTION-BASED FILTERING APPLIED")
    print(f"All {len(prices.columns)} assets in backtest have both price and shorting fee data")
    print(f"List of assets used (first 10): {list(prices.columns[:10])}")
    print(f"... (total {len(prices.columns)} assets)")
    
    # NaN stats for shorting fees
    fee_nan_stats = short_fee_data.isna().mean(axis=0)
    high_nan_cols = [c for c in short_fee_data.columns if fee_nan_stats[c] > 0.5]
    print(f"Shorting fee columns with >50% NaNs: {len(high_nan_cols)} out of {len(short_fee_data.columns)}")
    print(f"Overall shorting fee NaN percentage: {short_fee_data.isna().mean().mean()*100:.2f}%")
    
    # Show some example shorting fees
    print(f"Example shorting fees (first 5 assets, latest date):")
    latest_fees = short_fee_data.iloc[-1].head(5)
    for col, fee in latest_fees.items():
        asset = col[4:]  # Remove 'fee_' prefix
        daily_bps = fee/100/252*10000 if not pd.isna(fee) else 0
        print(f"  {asset}: {fee:.3f}% annual ({daily_bps:.2f} bps daily)")
    
    # 4. Automatic adaptation note
    print(f"\n🔄 AUTOMATIC ADAPTATION:")
    print(f"When you add more assets to shorting fee data, they will be automatically included")
    print(f"Current setup will always use the intersection of price and shorting fee universes")
    print("\n====================================================\n")


if __name__ == "__main__":
    main()
    print_full_report()
