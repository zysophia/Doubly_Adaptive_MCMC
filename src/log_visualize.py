import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as patches

palette ={"Kolmogorov": "darkblue", "parallelGibbs": "darkgreen", "superGibbs": "darkred", \
        "parallelGibbs (T$\leftarrow 1$)": "yellowgreen", "superGibbs (T$\leftarrow 1$)": "darkorange"}


def lineplot_complexity_vs_eps_Voting():
    df = pd.read_csv("../data/compare_complexity.csv")
    bmax = 0.1
    chain = "logical"
    weights = 7
    # df_sub = df[(df["chain"]==chain)&(df["beta_max"]==bmax)&(df["true_Q"]>1.38067)&(df["true_Q"]<1.38068)] # 1.37974, 1.38067, 
    df_sub = df[(df["chain"]==chain)&(df["beta_max"]==bmax)&(df["true_Q"]>1.37974)&(df["true_Q"]<1.37975)]
    df_sub = df_sub[(df_sub["algorithm"]=="Kolmogorov") | (df_sub["algorithm"]=="parallelGibbs")| (df_sub["algorithm"]=="superGibbs")]
    df_sub["1/$\epsilon$"] = 1/df_sub["epsilon"]
    df_sub["complexity"] = df_sub["sample_complexity"]
    plt.figure(figsize=(6,5))
    sns.set_theme(style="darkgrid")
    g = sns.lineplot(x="1/$\epsilon$", y="complexity", hue="algorithm", style = "algorithm", \
        ci=95 ,data=df_sub, markers=True, markersize = 8, dashes=False, linewidth=2, palette=palette) # style = "different_weights", 
    # g.set(ylim=(4, 11))
    g.set_yscale('log')
    g.set_xscale('log')
    plt.grid(True,which="both",ls="--",c='gray', alpha=0.5) 
    plt.show()
    g.figure.savefig("../figures/complexity_vs_eps_Voting.png", dpi=200)


def lineplot_complexity_vs_eps_Voting_zoom():
    df = pd.read_csv("../data/compare_complexity.csv")
    bmax = 0.1
    chain = "logical"
    weights = 7
    df_sub = df[(df["chain"]==chain)&(df["beta_max"]==bmax)&(df["true_Q"]>1.38067)&(df["true_Q"]<1.38068)] # 1.37974, 1.38067, 
    # df_sub = df[(df["chain"]==chain)&(df["beta_max"]==bmax)&(df["true_Q"]>1.37974)&(df["true_Q"]<1.37975)]
    df_sub = df_sub[df_sub["algorithm"]!="Kolmogorov"]
    df_sub["1/$\epsilon$"] = 1/df_sub["epsilon"]
    df_sub["algorithm"].replace({"parallelGibbs (McMcPro)": "parallelGibbs (T$\leftarrow 1$)", "superGibbs (McMcPro)": "superGibbs (T$\leftarrow 1$)"}, inplace=True)
    df_sub["complexity"] = df_sub["sample_complexity"]
    plt.figure(figsize=(10.5,6))
    
    sns.set_theme(style="darkgrid")
    plt.grid(True,which="both",ls="--",c='gray', alpha=0.5) 

    g = sns.lineplot(x="1/$\epsilon$", y="complexity", hue="algorithm", style = "algorithm", \
        ci=0 ,data=df_sub, markers=True, markersize = 7, dashes=False, linewidth=1.5, palette=palette) 
    # g.set(ylim=(4, 11))
    plt.legend(loc='lower right')
    ax2 = plt.axes([0.14, 0.45, .4, .49], facecolor='gainsboro')
    sns.lineplot(x="1/$\epsilon$", y="complexity", hue="algorithm", style = "algorithm", \
        ci=0 ,data=df_sub, markers=True, markersize = 7, dashes=False, linewidth=1.5, palette=palette, ax=ax2, legend=None) 
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set(xticklabels=[])  
    ax2.set(xlabel=None)
    ax2.set(yticklabels=[])  
    ax2.set(ylabel=None)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_xlim([170, 2200])
    ax2.set_ylim([10**6*1.6, 10**8*1.4])

    g.set_yscale('log')
    g.set_xscale('log')
    plt.grid(True,which="both",ls="--",c='gray', alpha=0.5) 

    min_f0, max_f0 = 170, 2200
    min_f1, max_f1 = 10**6*1.6, 10**8*1.4
    width = max_f0 - min_f0
    height = max_f1 - min_f1
    
    g.add_patch(
        patches.Rectangle(
            xy=(min_f0, min_f1), 
            width=width,
            height=height,
            linewidth=1,
            color='black',
            linestyle="dashed",
            fill=False
        )
    )
    plt.show()
    g.figure.savefig("../figures/complexity_vs_eps_Voting2.png", dpi=200)


def scatterplot_complexity_vs_Z_Voting():
    df = pd.read_csv("../data/compare_complexity.csv")
    epss = 0.025
    df_sub = df[(df["epsilon"]==epss) & (df["chain"]=="logical")]
    df_sub = df_sub[(df_sub["algorithm"]=="Kolmogorov") | (df_sub["algorithm"]=="parallelGibbs")| (df_sub["algorithm"]=="superGibbs")]
    df_sub = df_sub[(df_sub["true_Q"]<1.6) & (df_sub["true_Q"]>1.3)]
    nn = 5
    df_sub = df_sub[(df_sub["n"]==nn)]
    df_sub["1/Z"] = 1/(2**nn/df_sub["true_Q"])
    df_sub["complexity"] = df_sub["sample_complexity"]
    plt.figure(figsize=(4.8,5))
    sns.set_theme(style="darkgrid")
    g = sns.scatterplot(x="1/Z", y="complexity",  hue = "algorithm", style = "algorithm", ci=95 ,data=df_sub, palette=palette) # style = "different_weights", 
    g.set(xlim=(0.04, 0.05))
    g.set(ylim=(10**5, 10**8))
    g.set_yscale('log')
    g.set_xscale('log')
    g.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(0.02,1,0.02), numticks=10))
    g.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(0.1,1,0.1), numticks=10))
    plt.grid(True,which="both",ls="--",c='gray', alpha=0.5)  
    plt.show()
    g.figure.savefig("../figures/complexity_vs_Z.png", dpi=200)


def lineplot_complexity_vs_eps_Ising(n=6):
    df = pd.read_csv("../data/isingcompare_complexity.csv")
    chain = "Ising"
    df_sub = df[(df["chain"]==chain)&(df["n"]==n)]
    # df_sub["log10 (sample complexity)"] = df_sub["sample_complexity"]
    df_sub["1/$\epsilon$"] = 1/df_sub["epsilon"]
    df_sub["complexity"] = df_sub["sample_complexity"]
    plt.figure(figsize=(5.4, 6))
    sns.set_theme(style="darkgrid")
    g = sns.lineplot(x="1/$\epsilon$", y="complexity", hue="algorithm", style = "algorithm", \
        ci=95 ,data=df_sub, markers=True, markersize = 8, dashes=False, linewidth=2, palette=palette) # style = "different_weights", 
    # g.set(ylim=(10**5, 10**12))
    g.set_yscale('log')
    g.set_xscale('log')
    plt.grid(True,which="both",ls="--",c='gray', alpha=0.5)  
    plt.show()
    g.figure.savefig(f"../figures/complexity_vs_eps_Ising{n}.png", dpi=200)


def lineplot_error_vs_eps_Voting():
    df = pd.read_csv("../data/compare_complexity.csv")
    chain = "logical"
    n = 3
    epsilons = [0.1, 0.075, 0.05, 0.04, 0.025, 0.01]
    df_sub = df[(df["chain"]==chain)&(df["n"]==n)&(df["true_Q"]>1.38067)&(df["true_Q"]<1.38068)]
    df_sub = df_sub[df_sub["estimate_Q"]==df_sub["estimate_Q"]]
    df_sub = df_sub[(df_sub["algorithm"]=="Kolmogorov") | (df_sub["algorithm"]=="parallelGibbs")| (df_sub["algorithm"]=="superGibbs")]
    df_sub = df_sub[df_sub["epsilon"].isin(epsilons)]
    df_sub["absolute relative error"] = np.abs(df_sub["error"])
    df_sub["$\epsilon$"] = df_sub["epsilon"]
    plt.figure(figsize=(4.8,5))
    sns.set_theme(style="darkgrid")
    g = sns.lineplot(x="$\epsilon$", y="absolute relative error", hue="algorithm", style = "algorithm", \
        ci=35 ,data=df_sub, markers=True, dashes=False, palette=palette) # style = "different_weights", 
    # g.set(ylim=(5, 12))
    g.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    g.invert_xaxis()
    plt.grid(True,which="both",ls="--",c='gray', alpha=0.5)
    plt.show()
    g.figure.savefig("../figures/error_vs_epsilon.png", dpi=200)

if __name__ == "__main__":
    
    # lineplot_complexity_vs_eps_Voting()
    lineplot_complexity_vs_eps_Voting_zoom()
    # scatterplot_complexity_vs_Z_Voting()
    # lineplot_complexity_vs_eps_Ising(n=2)
    # lineplot_complexity_vs_eps_Ising(n=3)
    # lineplot_complexity_vs_eps_Ising(n=4)
    # lineplot_complexity_vs_eps_Ising(n=6)
    # lineplot_error_vs_eps_Voting()

   



    



    


