import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap
from scipy.special import expit
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import constants
import sys

VERBOSE = False
target_lang = None

task = sys.argv[1]
if task == "mttolang" or task == "mtfromlang":
    target_lang = sys.argv[2]

plt.rc("font", family="serif")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")


def gini(populations, accuracy):
    assert len(populations) == len(accuracy)
    N = len(populations)
    sum_nom = 0
    sum_denom = 0
    for i in range(N):
        for j in range(N):
            sum_nom += (
                populations[i] * populations[j] * np.abs(accuracy[i] - accuracy[j])
            )
        sum_denom += populations[i] * accuracy[i]
    return sum_nom / (2 * np.sum(populations) * sum_denom)


TOTAL_LANGS = 6500

if task == "inflection":
    all_populations = constants.read_sig_populations()
    languages = constants.get_sig_languages()
    languageso = constants.get_sig_languages()
    sys1 = "CULing-01-0"
    sys2 = "deepspin-02-1"
    sys3 = "uiuc-01-0"
    acc1 = constants.read_sig(system=sys1)
    acc2 = constants.read_sig(system=sys2)
    acc3 = constants.read_sig(system=sys3)
    populationso = [all_populations[l] for l in languages]
    accuracy1o = [acc1[l] for l in languages]
    accuracy2o = [acc2[l] for l in languages]
    accuracy3o = [acc3[l] for l in languages]
    accuracyo = [max(acc1[l], acc2[l], acc3[l]) for l in languages]
elif task == "tts":
    all_populations = constants.read_synthesis_populations()
    languages = constants.get_wilderness_languages()
    languageso = constants.get_wilderness_languages()
    all_bleus = constants.read_wilderness()
    # languageso.append('eng')
    languages.remove("alb")
    languages.remove("khi")
    languages.remove("may")
    languages.remove("nah")
    languageso.remove("alb")
    languageso.remove("khi")
    languageso.remove("may")
    languageso.remove("nah")
    populationso = [all_populations[l] for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
    # Needed because lower is better
    max_accuracy = max(accuracyo)
    min_accuracy = min(accuracyo)
    spread = max_accuracy - min_accuracy
    # print(min_accuracy, max_accuracy, spread)
    accuracyo = [(max_accuracy - a) / spread for a in accuracyo]
elif task == "xnli":
    all_populations = constants.read_xnli_populations()
    languages = constants.get_xnli_languages()
    languageso = constants.get_xnli_languages()
    all_bleus = constants.read_xnli_acc()
    populationso = [all_populations[l] for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "dep":
    all_populations = constants.read_dep_populations()
    languages = constants.get_dep_languages()
    languageso = constants.get_dep_languages()
    all_bleus = constants.read_dep(metric="las", system="udf")
    populationso = [all_populations[l] for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "qa":
    all_populations = constants.read_qa_populations()
    languages = constants.get_qa_languages()
    languageso = constants.get_qa_languages()
    all_bleus = constants.read_qa_acc()
    populationso = [all_populations[l] for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "topic":
    all_populations = constants.read_topic_populations()
    languages = constants.get_topic_languages()
    languageso = constants.get_topic_languages()
    all_bleus = constants.read_topic_acc()
    populationso = [all_populations[l] for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "senti":
    all_populations = constants.read_senti_populations()
    languages = constants.get_senti_languages()
    languageso = constants.get_senti_languages()
    all_bleus = constants.read_senti_acc()
    populationso = [all_populations[l] for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "averaged":
    all_populations = constants.read_averaged_populations()
    languages = constants.get_averaged_languages()
    languageso = constants.get_averaged_languages()
    all_bleus = constants.read_averaged_acc()
    populationso = [all_populations[l] for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "data":
    all_populations = constants.read_data_populations()
    languages = constants.get_data_languages()
    languageso = constants.get_data_languages()
    all_bleus = constants.read_data_acc()
    populationso = [all_populations[l] for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "sdqa_arabic":
    all_populations = constants.read_sdqa_arabic_populations()
    languages = constants.get_sdqa_arabic_languages()
    languageso = constants.get_sdqa_arabic_languages()
    all_bleus = constants.read_sdqa_arabic_acc()
    MSA_pop = all_populations["ara"]
    MSA_acc = all_bleus["ara"]
    languages.remove("ara")
    languageso.remove("ara")
    populationso = [all_populations[l] / 1000000 for l in languages]
    # accuracyo = [all_bleus[l]/MSA_acc for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "sdqa_swahili":
    all_populations = constants.read_sdqa_swahili_populations()
    languages = constants.get_sdqa_swahili_languages()
    languageso = constants.get_sdqa_swahili_languages()
    all_bleus = constants.read_sdqa_swahili_acc()
    SWA_pop = all_populations["swa"]
    SWA_acc = all_bleus["swa"]
    languages.remove("swa")
    languageso.remove("swa")
    populationso = [all_populations[l] / 1000000 for l in languages]
    # accuracyo = [all_bleus[l]/SWA_acc for l in languages]
    accuracyo = [all_bleus[l] for l in languages]
elif task == "mttoall":
    all_populations = constants.read_mt_populations()
    languages1 = constants.get_mt_languages()
    languages2 = constants.get_mt_languages()
    languageso = constants.get_mt_languages()
    all_bleus = constants.read_triangulated_BLEUs()
    for l in languages2:
        all_bleus[l, l] = 1
    populationso = [all_populations[l] for l in languages2]
    accuracyo = [
        np.average([all_bleus[l1, l2] for l1 in languages1]) for l2 in languages2
    ]
elif task == "mtfromall":
    all_populations = constants.read_mt_populations()
    languages1 = constants.get_mt_languages()
    languages2 = constants.get_mt_languages()
    languageso = constants.get_mt_languages()
    all_bleus = constants.read_triangulated_BLEUs()
    for l in languages2:
        all_bleus[l, l] = 1
    populationso = [all_populations[l] for l in languages2]
    accuracyo = [
        np.average([all_bleus[l2, l1] for l1 in languages1]) for l2 in languages2
    ]
elif task == "mtfromlang":
    all_populations = constants.read_mt_populations()
    languages1 = constants.get_mt_languages()
    languages2 = constants.get_mt_languages()
    languageso = constants.get_mt_languages()
    # languages1.remove(target_lang)
    # languages2.remove(target_lang)
    # languageso.remove(target_lang)
    all_bleus = constants.read_BLEUs(from_eng=True)
    all_bleus[target_lang, target_lang] = 100
    populationso = [all_populations[l] for l in languages2]
    accuracyo = [all_bleus[target_lang, l2] for l2 in languages2]
elif task == "mttolang":
    all_populations = constants.read_mt_populations()
    languages1 = constants.get_mt_languages()
    languages2 = constants.get_mt_languages()
    languageso = constants.get_mt_languages()
    # languages1.remove(target_lang)
    # languages2.remove(target_lang)
    # languageso.remove(target_lang)
    all_bleus = constants.read_BLEUs(to_eng=True)
    all_bleus[target_lang, target_lang] = 100
    print(all_bleus)
    populationso = [all_populations[l] for l in languages2]
    accuracyo = [all_bleus[l2, target_lang] for l2 in languages2]


# TOTAL_POPULATION = constants.TOTAL_SEA_LANG_SPEAKERS / 1000000
# pop_denom = constants.TOTAL_SEA_LANG_SPEAKERS / 1000000

TOTAL_POPULATION = np.sum(populationso)
pop_denom = TOTAL_POPULATION

if task == "sdqa_arabic":
    TOTAL_POPULATION = MSA_pop / 1000000
    pop_denom = TOTAL_POPULATION / 1000000
    TOTAL_LANGS = len(languages) + 2
elif task == "sdqa_swahili":
    TOTAL_POPULATION = SWA_pop / 1000000
    pop_denom = TOTAL_POPULATION / 1000000
    TOTAL_LANGS = len(languages) + 2

populationso2 = list(populationso)
languageso2 = list(languageso)
accuracyo2 = list(accuracyo)

sumpop = np.sum(populationso)
others = TOTAL_POPULATION - sumpop
print(others)
if task == "mttolang" or task == "mtfromlang":
    # others -= all_populations[target_lang]
    task2 = task + f"_{target_lang}"
populationso.append(others)
# languages.append('other')
languageso.append("other")
if task == "sdqa_arabic":
    accuracyo.append(0.55)
elif task == "sdqa_swahili":
    accuracyo.append(0.2)
else:
    accuracyo.append(0)

TOTAL_LANGS = len(languageso) + 1

remaining_langs = TOTAL_LANGS - len(languageso) + 1
pop_portion = others / float(remaining_langs)
populationso2 += [pop_portion] * remaining_langs
languageso2 += ["other"] * remaining_langs
accuracyo2 += [0] * remaining_langs


def include_diversity(l, T=1):
    acc_arr = np.array(l)
    acc_arr = [f**T for f in acc_arr]
    N = sum(acc_arr)
    acc_arr = [f / N for f in acc_arr]
    return list(acc_arr)

def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]


langs_to_show = set()

# temperatures = list(np.flip(np.arange(1,10)/10)) + [0.01]
temperatures = [0.01, 0.2, 0.3, 0.5, 0.7, 1.0]
# temperatures = [0.5]

for temperature in temperatures:
    if temperature == 1:
        accuracy = list(accuracyo)
        languages = list(languageso)
        populations = list(populationso)
    else:
        accuracy = list(accuracyo2)
        languages = list(languageso2)
        populations = list(populationso2)

    populations = include_diversity(populations, T=temperature)

    inds = np.flip(np.argsort(accuracy))
    populations = [populations[i] for i in inds]
    accuracy = [accuracy[i] for i in inds]
    languages = [languages[i] for i in inds]

    N = np.sum(populations)
    old_populations = [p / N for p in populations]

    populations = include_diversity(old_populations, T=temperature)
    gini_coeff = gini(np.array(populations) * N, accuracy)

    name = "Set1"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors * 200  # type: list

    temp = [0] + populations

    x = np.cumsum(temp)
    print("x", x)
    if task == "data":
        M = 500
    elif task == "averaged":
        for i in range(len(accuracy)):
            if languages[i] == "eng":
                M = accuracy[i]
    else:
        M = max(accuracy)
    # M = 1
    # M = max(accuracy)
    # M = 1000
    if VERBOSE:
        print(f"Max Accuracy: {M}")
    if M > 1:
        y = [a / M for a in accuracy]
    else:
        y = [a for a in accuracy]
    if VERBOSE:
        print(f"Simple macro-averaged accuracy: {np.average(y)}, {np.average(y)*M}")
        print(f"Gini Coefficient: {gini_coeff}")

    hsv_modified = get_cmap(
        "RdYlGn_r", 256
    )  # create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
    # newcmp = ListedColormap(hsv_modified(np.linspace(0, 1, len(y))))
    newcmp = ListedColormap(hsv_modified(np.linspace(0, 1, 100)))
    cmap = newcmp
    colors = cmap.colors

    def make_error_boxes(
        ax,
        xdata,
        ydata,
        lang,
        langs_to_show,
        facecolor="r",
        edgecolor="None",
        alpha=0.8,
    ):
        # Create list for all the error patches
        boxes = []
        area_covered = []
        area_missing = []
        addlangs = True
        if langs_to_show:
            addlangs = False

        ded_font = 7
        rot = 0
        props = {
            "ha": "left",
            "va": "bottom",
        }

        # ####
        # # (ORIGINAL) Priority based on quality
        # ####

        # # Loop over data points; create box from errors at each point
        # N = len(ydata)
        # for i in range(N):
        #     x0 = xdata[i]
        #     x1 = xdata[i + 1]
        #     y1 = ydata[i]
        #     area_covered.append(y1 * (x1 - x0))
        #     area_missing.append((1 - y1) * (x1 - x0))
        #     if y1 or lang[i] == "other":
        #         rect = Rectangle((x0, 0), x1 - x0, y1)
        #         # print(100-int(y1*100), lang[i],y1)
        #         if lang[i] == target_lang:
        #             pc = PatchCollection(
        #                 [rect],
        #                 facecolor=colors[100 - int(y1 * 100)],
        #                 alpha=0.5,
        #                 edgecolor=colors[100 - int(y1 * 100)],
        #                 hatch="//",
        #             )
        #             # pc = PatchCollection([rect], facecolor=colors[100-int(y1*100)], alpha=0.5, edgecolor=None, hatch='//')
        #         elif 100 - int(y1 * 100) == 100:
        #             # print("got here")
        #             # pc = PatchCollection([rect], facecolor=colors[99-int(y1*100)], alpha=alpha, edgecolor=colors[99-int(y1*100)])
        #             pc = PatchCollection(
        #                 [rect],
        #                 facecolor=colors[99 - int(y1 * 100)],
        #                 alpha=alpha,
        #                 edgecolor=None,
        #             )
        #         else:
        #             # pc = PatchCollection([rect], facecolor=colors[100-int(y1*100)], alpha=alpha, edgecolor=colors[100-int(y1*100)])
        #             pc = PatchCollection(
        #                 [rect],
        #                 facecolor=colors[100 - int(y1 * 100)],
        #                 alpha=alpha,
        #                 edgecolor=None,
        #             )
        #         ax.add_collection(pc)
        #         if lang[i] == "eng":
        #             # ax.text(x0, y1, f"{y1:.1f}", props, fontsize=ded_font, rotation=rot)
        #             ax.text(
        #                 x0 + 0.03, -0.15, lang[i], props, fontsize=ded_font, rotation=90
        #             )
        #         elif lang[i] == "other":
        #             # ax.text(
        #             #     x0 + (1 - x0) / 3,
        #             #     -0.1,
        #             #     lang[i],
        #             #     props,
        #             #     fontsize=ded_font,
        #             #     rotation=0,
        #             # )
        #             pass
        #         elif lang[i] in langs_to_show:
        #             # ax.text(x0, y1, f"{y1:.2f}"[1:], props, fontsize=ded_font, rotation=rot)
        #             # ax.text(x0+(x1-x0)/3, -0.12, lang[i], props, fontsize=ded_font, rotation=90)
        #             ax.text(x0, -0.15, lang[i], props, fontsize=ded_font, rotation=90)

        ####
        # (SEACrowd) Priority based on utility
        ####

        # Loop over data points; create box from errors at each point
        N = len(ydata)
        for i in range(N):
            x0 = xdata[i]
            x1 = xdata[i + 1]
            y1 = ydata[i]
            area_covered.append(y1 * (x1 - x0))
            area_missing.append((1 - y1) * (x1 - x0))

        # Sort by area_missing
        inds = np.flip(np.argsort(area_missing))

        # only show top 20
        x_data = [0]
        y_data = []
        lang_data = []
        x_prev = 0
        listlangs = []
        for i in inds[:20]:
            listlangs.append(lang[i])
            x0 = x_prev
            x1 = x_prev + (xdata[i + 1] - xdata[i])*9
            x_data.append(x1)
            y_data.append(ydata[i])
            lang_data.append(lang[i])
            x_prev = x1
        listlangs = ",".join(listlangs)
        print("listlangs", listlangs)

        x_data = normalize(
            x_data,
            {'actual': {'lower': np.min(x_data), 'upper': np.max(x_data)}, 'desired': {'lower': 0, 'upper': 1}}
        )

        for i in range(len(y_data)):
            # print(i, lang[i])
            x0 = x_data[i]
            x1 = x_data[i + 1]
            y1 = y_data[i]
            if y1 or lang_data[i] == "other":
                rect = Rectangle((x0, 0), x1 - x0, y1)
                # print(100-int(y1*100), lang_data[i],y1)

                # Add black to the missing area
                miss_rect = Rectangle((x0, y1), x1 - x0, 1)
                if lang_data[i] == target_lang:
                    pc = PatchCollection(
                        [rect],
                        facecolor=colors[100 - int(y1 * 100)],
                        alpha=0.5,
                        edgecolor=colors[100 - int(y1 * 100)],
                        hatch="//",
                    )
                    # pc = PatchCollection([rect], facecolor=colors[100-int(y1*100)], alpha=0.5, edgecolor=None, hatch='//')
                elif 100 - int(y1 * 100) == 100:
                    # print("got here")
                    # pc = PatchCollection([rect], facecolor=colors[99-int(y1*100)], alpha=alpha, edgecolor=colors[99-int(y1*100)])
                    pc = PatchCollection(
                        [rect],
                        facecolor=colors[99 - int(y1 * 100)],
                        alpha=alpha,
                        edgecolor=None,
                    )
                else:
                    # pc = PatchCollection([rect], facecolor=colors[100-int(y1*100)], alpha=alpha, edgecolor=colors[100-int(y1*100)])
                    pc = PatchCollection(
                        [rect],
                        facecolor=colors[100 - int(y1 * 100)],
                        alpha=alpha,
                        edgecolor=None,
                    )
                miss_pc = PatchCollection(
                    [miss_rect],
                    facecolor="#333333",
                    alpha=0.7,
                    # edgecolor="#000000",
                    hatch="//",
                )
                ax.add_collection(pc)
                ax.add_collection(miss_pc)
                
                if lang_data[i] == "eng":
                    # ax.text(x0, y1, f"{y1:.1f}", props, fontsize=ded_font, rotation=rot)
                    ax.text(
                        x0 + 0.03, -0.2, lang_data[i], props, fontsize=ded_font, rotation=90
                    )
                    ax.text(
                        x0, -0.35, str(i + 1), props, backgroundcolor='#700701', color="#ffffff", fontsize=ded_font-2, rotation=90
                    )
                elif lang_data[i] == "other":
                    # ax.text(
                    #     x0 + (1 - x0) / 3,
                    #     -0.1,
                    #     lang_data[i],
                    #     props,
                    #     fontsize=ded_font,
                    #     rotation=0,
                    # )
                    pass
                elif lang_data[i] in langs_to_show:
                    # ax.text(x0, y1, f"{y1:.2f}"[1:], props, fontsize=ded_font, rotation=rot)
                    # ax.text(x0+(x1-x0)/3, -0.12, lang_data[i], props, fontsize=ded_font, rotation=90)
                    ax.text(x0, -0.25, lang_data[i], props, fontsize=ded_font, rotation=90)
                    ax.text(
                        x0, -0.45, str(i + 1), props, backgroundcolor='#a60f0f', color="#ffffff", fontsize=ded_font-1, rotation=0, fontweight='bold',
                    )
        # ax.text(0.4, 0.9, f"$\\tau = {temperature}$", props, fontsize=ded_font+4, rotation=0)
        return area_covered, area_missing

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(8, 2))
    ax.set_prop_cycle(color=colors)

    langs_to_show = set()
    listlangs = "deu,cmn,eng,ell,spa,hin,tam,ben,lin,kor,por,other"
    if task == "qa":
        listlangs = ",".join(languages)
    elif task == "topic":
        listlangs = ",".join(languages)
    elif task == "senti":
        listlangs = ",".join(languages)
    elif task == "averaged":
        listlangs = ",".join(languages)
    elif task == "data":
        listlangs = ",".join(languages)
    elif task == "sdqa_arabic":
        temp = list(languages)
        temp.remove("afb")
        listlangs = ",".join(temp)
    elif task == "sdqa_swahili":
        listlangs = "KN,TZ,other"
    elif task == "xnli":
        listlangs = "deu,cmn,eng,fin,spa,hin,tam,tgl,ben,lin,other"
    elif task == "tts":
        listlangs = "deu,cmn,eng,ell,spa,hin,tam,ben,lin,aka,mal,other"
    elif task == "inflection":
        listlangs = "deu,eng,ell,spa,hin,tam,fin,tgl,ben,lin,other"
    elif task == "dep":
        listlangs = "ces,deu,cmn,eng,ell,spa,hin,tam,fin,tgl,ben,lin,amh,others"
    elif (task == "mttolang" or task == "mtfromlang") and target_lang == "eng":
        listlangs = "eng,ljl,ace,ban,bjn,bug,ceb,ilo,ind,jav,kac,khm,lao,lus,min,mya,pag,shn,sun,tha,vie,war,zsm,fil,zlm,hmv,bbc,nij,mad"
    for lll in listlangs.split(","):
        langs_to_show.add(lll)
    if task == "mttolang" or task == "mtfromlang":
        langs_to_show.add(target_lang)

    if task == "sdqa_arabic":
        # ax.hlines(1,0,1,color='k',linestyles='dashed')
        ax.plot([0, 1], [MSA_acc, MSA_acc], "k--", linewidth=0.5)
        ax.text(
            0.1,
            MSA_acc,
            "(Written) Modern Standard Arabic",
            {
                "ha": "left",
                "va": "bottom",
            },
            fontsize=7,
            rotation=0,
        )
    elif task == "sdqa_swahili":
        # ax.hlines(1,0,1,color='k',linestyles='dashed')
        ax.plot([0, 1], [SWA_acc, SWA_acc], "k--", linewidth=0.5)
        ax.text(
            0.1,
            SWA_acc,
            "(Written) Coastal Swahili",
            {
                "ha": "left",
                "va": "bottom",
            },
            fontsize=7,
            rotation=0,
        )

    area_covered, area_missing = make_error_boxes(ax, x, y, languages, langs_to_show)

    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=True,
        labelleft=True,
    )  # labels along the bottom edge are off

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    if task == "sdqa_arabic":
        ax.set_xlabel("Number of Arabic Speakers", fontsize=9, labelpad=20)
    elif task == "sdqa_swahili":
        ax.set_xlabel("Number of Swahili Speakers", fontsize=9, labelpad=20)
    else:
        ax.set_xlabel(f"# Speakers for SEA Languages ($\\tau = {temperature}$)", fontsize=9, labelpad=40)
    if task == "averaged":
        ax.set_ylabel("Model Capability", fontsize=9)
    elif task == "data":
        ax.set_ylabel("Data Availability", fontsize=9)
    else:
        ax.set_ylabel("Rel. Quality", fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    tmpr = f"{temperature}".replace(".", "_")
    if task == "mtfromlang" or task == "mttolang":
        plt.savefig(f"figs/final/{task2}_{tmpr}.pdf", pad_inches=0)
    else:
        plt.savefig(f"figs/final/{task}_{tmpr}.pdf", pad_inches=0)
    plt.show()

    print(f"{target_lang}\tM={sum(area_covered)}")
    # print(f"{target_lang} total area covered: M={sum(area_covered)}")
    # print(f"Total area missing: RoI={sum(area_missing)}")

    """
    inds = np.flip(np.argsort(area_covered))
    print(f"Top 10 Covered with tau = {temperature}")
    for i in inds[:10]:
        print(f"{i}\t{languages[i]}\t{area_covered[i]}\t{area_missing[i]}")
    """

    inds = np.flip(np.argsort(area_missing))
    print(f"\tTop 3 Missing with tau = {temperature}")
    for i in inds[:4]:
        print(f"\t{i}\t{languages[i]}\t{area_covered[i]}\t{area_missing[i]}")

    # print(langs_to_show)