##
# Fairseq are one file per reference, need to put all candidates and references in a single file to run BERTScore

import os

HOME = '/your/root'
HOMEDIR_REFS = '' #folder where fairseq references are
HOMEDIR_CANDIS = '' #folder where fairseq outputs are

#references should have this nb of instances
REFS = [('film', 'film_test', 2813),
                ('animal', 'animal_test', 2573),
                ('company', 'company_test', 2846)]

#outputs folders by selected models
CANDIS = {
    'animal': ['fconvdpp_prevl_iwslt_de_en_tdtk_r2r_animal_L800_checkpoint18',
               'fconv_iwslt_de_en_covloss__tdtk_r2r_animal_L800_checkpoint16',
               'fconv_iwslt_de_en_tdtk_r2r_animal_L800_checkpoint15'],

    'film': ['fconvdpp_prevl_iwslt_de_en_tdtk_r2r_film_L800_checkpoint14',
             'fconv_iwslt_de_en_covloss__tdtk_r2r_film_L800_checkpoint14',
             'fconv_iwslt_de_en_tdtk_r2r_film_L800_checkpoint14'],

    'company': ['fconvdpp_prevl_iwslt_de_en_tdtk_r2r_company_L800_checkpoint13',
                'fconv_iwslt_de_en_covloss__tdtk_r2r_company_L800_checkpoint11',
                'fconv_iwslt_de_en_tdtk_r2r_company_L800_checkpoint14']
}

for dataset, refdir, nbinst in REFS:

    print("Formatting... " + dataset)

    refpath = os.path.join(HOME, HOMEDIR_REFS, refdir)
    outref = open(os.path.join(HOME, HOMEDIR_REFS, refdir + '_all.ref'), 'w')
    datasetCandis = {}
    for modelOut in CANDIS[dataset]:
        datasetCandis[modelOut] = open(os.path.join(HOME, HOMEDIR_CANDIS, modelOut, 'all.dec'), 'w')

    for f in os.listdir(refpath):
        curf = open(os.path.join(refpath, f), 'r')
        outref.write(curf.readline() + '\n')
        for modelOut in CANDIS[dataset]:
            curmodf = open(os.path.join(HOME, HOMEDIR_CANDIS, modelOut, f.replace('.ref', '.dec')), 'r')
            datasetCandis[modelOut].write(curmodf.readline() + '\n')

    for k in datasetCandis.keys():
        datasetCandis[k].close()


