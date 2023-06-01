import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import locale


def winner_ev_calculator(stake: np.array,
                         prob_win: np.array,
                         deduction_mult: np.array,
                         fractional_price: np.array,
                         prob_lose: np.array) -> np.array:
    return stake * ((prob_win * deduction_mult * fractional_price) - prob_lose)


def place_ev_calculator(stake: np.array,
                        prob_win: np.array,
                        deduction_mult: np.array,
                        fractional_price: np.array,
                        prob_lose: np.array,
                        each_way_reduction: np.array,
                        each_way_bet: np.array) -> np.array:
    place_ev = np.where(each_way_bet == 0, 0,
                        stake * ((prob_win * deduction_mult * each_way_reduction * fractional_price) - prob_lose))

    return place_ev


def settle_winner_selection(
        stake: np.array,
        fractional_price: np.array,
        deduction_mult: np.array,
        won: np.array) -> np.array:
    profit = np.where(won == 1, stake * fractional_price * deduction_mult, -stake)

    return profit


def settle_place(
        stake: np.array,
        fractional_price: np.array,
        deduction_mult: np.array,
        placed: np.array,
        each_way_reduction: np.array,
        each_way_bet: np.array) -> np.array:
    profit = np.where(each_way_bet == 0, 0, np.where(placed == 0, -stake,
                                                     stake * fractional_price * deduction_mult * each_way_reduction))

    return profit


def simulate_race(
        win_probs: np.array,
        place_probs: np.array,
        races_to_sim: int) -> (np.array, np.array):
    # create winner and place random arrays
    win_rand = np.random.rand(races_to_sim)
    place_rand = np.random.rand(races_to_sim)

    # Simulate if wins
    win_result = np.where(win_rand < win_probs, 1, 0)

    # If won then has to place, if not won simulate if it places
    place_result = np.where(win_result == 1, 1, np.where(place_rand < place_probs, 1, 0))

    return win_result, place_result


if __name__ == "__main__":
    # ---------------------------------------------- Cleaning --------------------------------------------------------
    # read in data
    df = pd.read_csv("runner_bets.csv")

    # Isolate any problematic indexes
    no_probability = df[(df["BSP"] < 1) | ((df["EW"] == 1) & (df["BSPplace"] < 1))].index
    invalid_price = df[df["PriceTaken"] < 1].index
    invalid_deduction = df[df["Deduction"] > 1].index
    invalid_stakes = df[df["Stake"] < 0].index
    invalid_terms = df[(df["EW"] == 1) & (df["Terms"] < 1)].index
    won_no_place = df[(df["Winner"] == 1) & (df["Placed"] != 1)].index

    indexes_to_drop = np.unique(np.concatenate((no_probability, invalid_price, invalid_deduction,
                                                invalid_stakes, invalid_terms, won_no_place), 0))

    # Drop the above from our working df and reset the index incase we want to iterate through by index later
    df = df.drop(indexes_to_drop).reset_index()

    # Just drop the remaining na values I haven't caught with other cleaning. only 167 of them
    df = df.dropna()

    # Create some new columns to make future calculations simpler (fewer (x - 1) or 1/x calculations)
    df["ProbWin"] = 1 / df["BSP"]
    df["ProbPlace"] = 1 / df["BSPplace"]
    df["ProbNotWin"] = 1 - df["ProbWin"]
    df["ProbNotPlace"] = 1 - df["ProbPlace"]
    df["FractionalPrice"] = df["PriceTaken"] - 1
    df["DeductionMult"] = 1 - df["Deduction"]
    df["EachWayMult"] = np.where(df["Terms"].values != 0, 1 / df["Terms"], 1)

    # note we're assuming that the each-way stake is actually half of total bet. e.g. Stake = £10..
    # £10 on win, £10 on place -> total stake = £20

    # -------------------------------------------------Q1 Part 1------------------------------------------------------
    # ToDo rework the EV/settle functions to be more logical... probably split into EW / winner selections,
    #  handle separately then merge back together or just a handler-type function that calls the other two where
    #  necessary
    # Predicted EV based on ExpROI
    df["Predicted_EV"] = df["ExpROI"].values * df["Stake"].values

    ew_indexes = df[(df["EW"] == 1) & (df["Winner"] == 1)].index

    df["EV"] = winner_ev_calculator(df["Stake"].values, df["ProbWin"].values, df["DeductionMult"].values,
                                    df["FractionalPrice"].values, df["ProbNotWin"].values)

    df["EV"] += place_ev_calculator(df["Stake"].values, df["ProbPlace"].values, df["DeductionMult"].values,
                                    df["FractionalPrice"].values, df["ProbNotPlace"].values, df["EachWayMult"].values,
                                    df["EW"].values)

    df["PnL"] = settle_winner_selection(df["Stake"].values, df["FractionalPrice"].values, df["DeductionMult"].values,
                                        df["Winner"].values)

    df["PnL"] += settle_place(df["Stake"].values, df["FractionalPrice"].values, df["DeductionMult"].values,
                              df["Placed"].values, df["EachWayMult"].values, df["EW"].values)

    # get the cumulative PnL and EV
    df["cum_PnL"] = df["PnL"].cumsum()
    df["cum_EV"] = df["EV"].cumsum()

    # aggregate the last cumulative PnL and EV for each day
    df_PnL = df.groupby(pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y'))['cum_PnL'].agg(['last']).reset_index()
    df_EV = df.groupby(pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y'))['cum_EV'].agg(['last']).reset_index()

    # compute the two-week rolling average
    df_PnL["2_week_rolling"] = df_PnL["last"].rolling(14).mean()
    df_EV["2_week_rolling"] = df_EV["last"].rolling(14).mean()

    # plot results
    plt.figure(figsize=(12, 5))

    sns.lineplot(x="Date", y="last", data=df_PnL, label="Daily Profit")
    sns.lineplot(x="Date", y="2_week_rolling", data=df_PnL, label="Two week rolling profit")
    sns.lineplot(x="Date", y="last", data=df_EV, label="Daily EV")
    sns.lineplot(x="Date", y="2_week_rolling", data=df_EV, label="Two week rolling EV")

    x_pos = ["04/01/2021", "05/01/2021", "06/01/2021", "07/01/2021", "08/01/2021", "09/01/2021", "10/01/2021",
             "11/01/2021"]
    x_lab = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
    plt.xticks(x_pos, x_lab)

    y_pos = [-500000, 0, 500000, 1000000, 1500000, 2000000, 2500000]
    y_lab = ["-0.5", "0", "0.5", "1", "1.5", "2", "2.5"]
    plt.yticks(y_pos, y_lab)

    plt.xlabel("Months of the year 2021")
    plt.ylabel("Profit (millions)")

    # plt.show()

    # -------------------------------------------------Q1 Part 2------------------------------------------------------
    # set number of sims for monte-carlo and number of races to sim
    num_sims = 1
    num_races = len(df.index)

    # compute probability of a horse to place and not win
    df["PlaceNotWin"] = df["ProbPlace"] - df["ProbWin"]

    # set up the array to store profits from sims
    sim_profits = np.zeros(num_sims)

    # seed rng for reproducibility
    np.random.seed(1)

    for i in range(num_sims):
        # get simulated race results
        winners, placers = simulate_race(df["ProbWin"].values, df["PlaceNotWin"].values, num_races)

        # compute profits
        profits = settle_winner_selection(df["Stake"].values, df["FractionalPrice"].values, df["DeductionMult"].values,
                                          winners)
        profits += settle_place(df["Stake"].values, df["FractionalPrice"].values, df["DeductionMult"].values, placers,
                                df["EachWayMult"].values, df["EW"].values)
        # add profit to array
        sim_profits[i] = profits.sum()

    ev = sim_profits.mean()

    print(" -- Q1 -- \n")

    print("model assumed EV =   ", '${:,.2f}'.format(df["Predicted_EV"].sum()))
    print("calculated EV =      ", '${:,.2f}'.format(df["EV"].sum()))
    print("simmed EV =          ", '${:,.2f}'.format(ev))
    print("\nModel asummed EV based off ROI is obviously wrong. \nI believe simmed and calculated should be the same"
          " so probably something wrong in my calculations of one of them.. come back to this\n")

    # ----------------------------------------------------Q2----------------------------------------------------------
    # ---- a ----

    # note that whilst I had to do some cleaning at the start of Q1... the expROI from the model just seems
    # ridiculous. There's ~19,000 ROI predictions of either 2.0 or 3.0 that have no correlation to the BSP and Price

    # Calculate actual ROI for each bet
    df["Actual_ROI"] = df["EV"].values / df["Stake"].values

    # Filter out the silly ROIs
    sensible_df = df[df["ExpROI"] < 1.0].reset_index()

    # Look at the silly ROIs
    silly_df = df[df["ExpROI"] >= 1.0].reset_index()

    actual_roi_average = round(df["Actual_ROI"].mean(), 4)
    general_roi_average = round(df["ExpROI"].mean(), 4)
    silly_roi_average = round(silly_df["ExpROI"].mean(), 4)
    sensible_roi_average = round(sensible_df["ExpROI"].mean(), 4)

    # Do a pearson correlation test
    # General
    general_pearson = pearsonr(df["ExpROI"].values, df["Actual_ROI"].values)
    silly_pearson = pearsonr(silly_df["ExpROI"].values, silly_df["Actual_ROI"].values)
    sensible_pearson = pearsonr(sensible_df["ExpROI"].values, sensible_df["Actual_ROI"].values)

    pearson_stats = np.around([general_pearson[0], silly_pearson[0], sensible_pearson[0], np.nan], 2)
    pearson_probs = np.around([general_pearson[1], silly_pearson[1], sensible_pearson[1], np.nan], 4)
    mean_rois = [general_roi_average, silly_roi_average, sensible_roi_average, actual_roi_average]

    df_2a = pd.DataFrame(data=[pearson_stats, pearson_probs, mean_rois],
                         index=["Pearson statistic", "Pearson p-value", "Mean ROI"],
                         columns=["Model", "Model (ExpROI>1)", "Model (ExpROI<1)", "Actual"])

    print(" -- Q2 --")
    print(" - a - \n")
    print("In general the model is a poor prediction of actual ROI. \nBut this is mainly because of ~19,000 ExpROI "
          "values of either 2.0 or 3.0. If we filter these out any ROIs greater than 1.0 in the model\nit actually "
          "has a weak positive correlation with actual ROI as shown below.\n")
    print(df_2a)


    # ---- b ----
    # group mean Actual_ROI by hour
    times = pd.DatetimeIndex(df["Time_Placed"])
    grouped = df.groupby([times.hour]).Actual_ROI.mean()

    # pop into a df
    time_df = pd.DataFrame({"Hour": grouped.index, "Mean_ROI": grouped.values})

    # compute pearson
    time_pearson = pearsonr(time_df["Hour"].values, time_df["Mean_ROI"].values)

    print("\n - b - \n")
    print("We find that the later in the day we place a bet the lower the ROI as shown by the pearson statistics:")
    print("Pearson statistic:", round(time_pearson[0], 4) , "p-value:", round(time_pearson[1], 4))

    # ToDo: Comeback and maybe go more granular and look at ROI as function of time difference between bet placed and
    #  start of race

    # ---- c ----
    # Pearson correlation for stake
    stake_pearson = pearsonr(df["Stake"].values, df["Actual_ROI"].values)
    print("We find no obvious initial correlation between stake size and ROI")
    print("Stake Pearson correlation coefficient:", round(stake_pearson[0], 4), "p-value: ", round(stake_pearson[1], 4))

    # ToDo come back and maybe take a more granular look... maybe group the stakes then have a look
    # Note that obviously EV and PnL scale with stake, so looking at Actual ROI makes the most sense.
    quantiles = np.linspace(0, 1, 11)
    stakes = np.quantile(df["Stake"], quantiles)
    stakes_df = pd.DataFrame({"quantiles": quantiles, "stakes": stakes})
    print(stakes_df)

    # ----------------------------------------------------Q3----------------------------------------------------------
    # Whilst Profit is useful initially, given we are assuming the true probabilities are the betfair prices, it makes
    # sense just to rank order them by 3 metrics:
    # Want them to bet on +EV bets -> mean ROI
    # Consistency in stake sizes relative to EV of the bet -> variance/stddev in EV is a good metric for this
    # Volume -> number of bets

    # Create groupBy object
    grouped_by_runner = df.groupby("Runner")

    # Extract stats on each runner from the groupBy
    grouped_df = round(grouped_by_runner[["PnL"]].sum())
    grouped_df["ROI_Mean"] = round(grouped_by_runner[["Actual_ROI"]].mean() * 100, 1)
    grouped_df["EV_stddev"] = round(grouped_by_runner[["EV"]].std())
    grouped_df["NumBets"] = grouped_by_runner[["Runner"]].count()

    # drop anyone with less than, say, 50 bets as we don't know enough
    grouped_df = grouped_df[grouped_df["NumBets"] > 50]

    # now drop anyone with a negative mean ROI (on average they make bad bets)
    grouped_df = grouped_df[grouped_df["ROI_Mean"] > 0]

    # we could constrain by EV stddev, but would feel relatively arbitrary and a staking strategy could easily be
    # recommended

    # rank order the runners
    grouped_df.sort_values(by=["ROI_Mean", "EV_stddev", "NumBets"], inplace=True, ascending=[False, True, False])

    print("\n -- Q3 -- \n")
    print("The runners I would keep in order of their value to me:")
    print(grouped_df.index.values)
