import pandas as pd


def evaluate(df):
    agreed = df[df["Agreement?"] == "Yes"]
    compromised = df[df["Agreement?"] == "No"]

    ret = pd.concat([agreed["Decided"].value_counts(),
                     compromised["Decided"].value_counts(),
                     df["Decided"].value_counts()], axis=1)

    ret.set_axis(["unanimous", "compromised", "final count"], axis="columns", inplace=True)

    ret.loc["total"] = [ret["unanimous"].sum(), ret["compromised"].sum(), ret["final count"].sum()]
    ret = ret.fillna(0)
    return ret.to_latex()


def main():
    df = pd.read_csv("./kappa-pp.tsv", sep="\t")

    print(evaluate(df))
    input()

    print(evaluate(df.loc[:51]))
    input()

    print(evaluate(df.loc[52:102]))
    input()

    print(evaluate(df.loc[103:]))
    input()


if __name__ == '__main__':
    main()
