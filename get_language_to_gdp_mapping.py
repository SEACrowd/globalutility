import constants
import numpy as np

languages = constants.get_all_languages()
all_populations = constants.read_all_populations()

try:
    languages.remove("alb")
    languages.remove("khi")
    languages.remove("may")
    languages.remove("nah")
    # languages.remove('uig')
    # languages.remove('epo')
    # languages.remove('nno')
except:
    pass


demand1 = constants.read_gdp(languages)
ec_dem1 = [demand1[l] for l in languages]

populations, existing_languages = [], []
for l in languages:
    try:
        populations.append(all_populations[l])
        existing_languages.append(l)
    except:
        print(f"Population not found for {l}")

with open("economic_indicators_data/languages_to_gdp.tsv", "w") as op:
    op.write("ISO\tPopulation (million)\tGDP\n")
    for i, l in enumerate(existing_languages):
        op.write(f"{l}\t{populations[i]}\t{ec_dem1[i]}\n")
