001 - Vari-Cozi - Alexande Sassi @ 11:20
- most varient lie in non-coding region
- You can do Machenis Insights and Varient effect prediction
- Problems are with models: (Emformer - S2F understand promotor but not enhancer)
- Constraining teaining on varient simproves prediction of differences.
- Deep-Allele improves for allelic ratios. 
- SAGE-net trains on personal genomes. ( P-SAGE cannot improve predictions for unseen genes)
- Training on sequence varinet is promissing.


002 - Combining massively parallel reporter assays and graph genomics to assay the regulatory effects of indels and structural variants
- SuRE ()
- They used the Graph Genome. 
- SuRE-SV pipeline in nextflow.
- 

003 - Multilingual model improves zero-shot prediction of disease effects on proteins @ 11:2 1
Models for mutation effect prediction in coding sequences rely on sequence-, structure-, or homology-based features. Here, we introduce a novel method that combines a codon language model with a protein language model, providing a dual representation for evaluating effects of mutations on disease. By capturing contextual dependencies at both the genetic and protein level, our approach achieves a 3% increase in ROC-AUC classifying disease effects for 137,350 ClinVar missense variants across 13,791 genes, outperforming two single-sequence-based language models. Obviously the codon language model can uniquely differentiate synonymous from nonsense mutations at the genomic level. Our strategy uses information at complementary biological scales (akin to human multilingual models) to enable protein fitness landscape modeling and evolutionary studies, with potential applications in precision medicine, protein engineering, and genomics.
- Boden Lab
- USED SNV from ClinVar
- Used CalM and ESM - 2
- CLM-better and PLM- Better (Mutation changes and amono acid from high codon to low codon usage)
- Used Chi Square test.
- 

003 - X-MAP - Marco Anteghini
- Genetic variants, particularly missense mutations, can significantly affect protein function and contribute to disease development. Methods like CADD and AlphaMissense are widely used for pathogenicity prediction; however, their integration into existing resources remains limited due to compatibility issues and high computational demands.
We introduce X-MAP, an integrated platform that leverages protein language models to enhance variant effect prediction through a novel embedding-based strategy. This approach captures both local and global protein features, enabling more accurate interpretation of mutation impacts.
Our method generates embeddings for entire protein sequences using multiple state-of-the-art models—ESM2, ESMC, and ESM1v—and extracts contextual information around mutation sites using a dynamic window of four residues on each side. This window size was empirically optimized to balance detailed local structure with computational efficiency
We evaluated both concatenation and difference-based embedding strategies using rigorous 10-fold cross-validation with XGBoost classifiers on a large dataset of 71,595 genetic variants across 12,666 human proteins. Among all methods, the ESMC concatenation strategy with the 4-residue window achieved the highest performance (Accuracy: 0.84, MCC: 0.66, AUC: 0.90), outperforming the Esnp baseline (Accuracy: 0.82, MCC: 0.64, AUC: 0.82), which relies on full sequence concatenation.
By concentrating on regions directly affected by mutations while retaining global sequence context, X-MAP achieves both accuracy and computational efficiency. We are currently developing a hybrid Transformer-CNN model to further enhance prediction accuracy and interpretability. X-MAP represents a powerful and scalable framework for variant analysis with direct applications in precision medicine and disease research.

- Idea is to create mutaiton analysis platform.
- Integeration of protein language models.
- Validation to real world integeration
- ESM-C Model 300M 600M 6B Parameters.
- Used 3 datasets (ESNP&GO, X-MAP, X-MAP B&P)
- Concatinating several models can improve the performance so they applied two different frameworks.
- Combine two vectors  OR Difference of two vecotrs (delta)
- Best window for clasification of pathogenic and benign varient was 3 window sise.
- ESM-C reach AUC of ~0.88
- No questions as delayed. 

004 - StructGuy - Alexender
In recent years, machine learning models for predicting variant effects on protein function have been dominated by unsupervised models doing zero-shot predictions on the task. In their development, multiplexed assay of variant effect (MAVE) data played only a secondary role used for model evaluation, most prominently applied in the ProteinGym benchmark. Yet, the rapidly increasing amount of available MAVE data should be able to fuel novel supervised predictions models, but is hindered by data leakage resulting when MAVE data is used to train a supervised model. Such models are not able to generalize their predictions to proteins not present in the training data, hence so far they are only used in protein design tasks.
Here, we present the novel random forest-based prediction method StructGuy that overcomes the problem of data leakage by applying sophisticated splits in hyperparameter optimization and feature selection. By removing proteins similar to any proteins in our training data set from ProteinGym, we constructed a dedicated benchmark that aims to evaluate the ability of a supervised model to generalize to proteins not seen in the training data. In this benchmark, we could do a direct and fair comparison of our StructGuy model with all models that are part in the zero-shot substitutions track of ProteinGym, and were able to demonstrate a slightly higher average Spearmans' correlation coefficient (0.45 vs. second highest: ProtSSN: 0.43). StructGuy directly applied on ProteinGym results in an average Spearmans' correlation coefficient of 0.6.
- Clasical models are not so much differnet then new models. (Varient effect prediction in age of AI new review) 
- Used MAVEDB and ProteinGYM
- Centeral Idea: Model that can perform equally well without training data. 
- They have developed this method StructMan (Random Forest)
- Most important feature is Structure 
- Compare unsupervised model in the benchmarks.
- Split proteins from ProteinGYM so that we removed (Smae, clse, Distant and unrelated)


005 - Functional characterization of standing variation around disease-associated genes using Massively Parallel Reporter Assays - Killian
A substantial reservoir of disease-associated variants resides in non-coding sequences, particularly in proximal and distal gene regulatory sequences. As part of the NIH Impact of Genomic Variation on Function (IGVF) consortium, we investigated functional genetic variation using Massively Parallel Reporter Assays (MPRAs). We tested >28,000 candidate cis-regulatory regions (cCREs) in the proximity (50kb) of 526 neural, cardiac or clinically actionable genes as well as a random gene set. Within these cCREs, we included >46,000 variants from gnomAD. This included all single nucleotide variants (SNVs) with allele frequency (AF) ≥1% as well as 35,000 rare and singleton variants. Rare variants were prioritized using Enformer (Avsec Ž et al. 2021) to select 70% potentially activating, 15% repressing, and 15% random variants.
Performing this MPRA in NGN2-derived neurons from WTC-11 cells showed that 16% (4045) of cCREs have significantly different activity from negative controls, while 6% (1540) of elements exhibit distinct activity from scrambled controls (dCREs). Among the dCREs, 3.3% are significantly more active and 2.7% were less active. About 3% (1304) of the tested variants showed a significant allelic effect. We observed both common and rare variants with significant allelic effects, with rare variants contributing the larger proportion. Examples of significant common and singleton SNVs include rs11635753 and rs1257445811 affecting SMAD3 and TRIO, respectively, and both associated with complex neurological phenotypes. Using Enformer for prioritization resulted in an enrichment in the selected rare variants but also failed to effectively capture regulatory grammar at base resolution.
- Looked for diseases with cardiac and neuro.
- Looked into Screen database (database with varient tracks).
- 5% show significant activity. 
- 24000 varients and 37000 varients.
-


006 - Variant Interpretation at Scale, for safer and more effective disease treatment 
- Investigating PGx Varients in 1000,000 genomes Projects, NUDT15 protein.
- Varient Page has 6.5M Rare and Common Varients.
- Chellanges that might be addresable by AI?
- Project, Developement of Digital Twons for Rare Disease.
- Perturbation Catelogue. 

007 - Genetic Cofounding Hadasa 
- PRS = Pathogenetic Risk Score
- casual relationshops between diseases under genetic cofounders.
- Two Phenotypes 1. Dimentia and Hyperlipidemia (same Gene Apoe)
- What does the DAG look likem what is the reason/ 
- make a cohort and see isf the is high comorbity or low comorbidity? 
- Removed the people who have atleast 1 SNP.
- Why they did it twice? (vascular demientia and other demenia)
- Did not get significant results. 
- They started to think differently.
- They talked about Matching attenuates feature bias in casual inferance. 
- They tried for 12 diseases and found casual relations graphs.
- They found opposite direction, Parkinson -> Skitschophrania.


008 - nf-core @ 3:03 pm
- Nextflow is most widely used system in 2024 
- nf-core best practices and it can also make your pipeline impprtal.
- 83 pipeline, 41 are under developement 
- 1567 Modules (Wraper of Bioinformatic Tools)
- How you can contribute? Any Level.
- EPI2ME series of workflow, using nf-core template and are using their own models.
- Tree - Life: They are using nf-core modules, and they communicate back to community. (Euro FAANg)
- nf-core is standard for BovReg pipelines.
- Empowering Bioinformatics with nf-core pipelines a paper in BioRxiv.
- Upcoming Events: nf-core Hackathron 28th - 29th October.

# 009 - Splice Transformer - vari - Ning Shen
- Assistant professor works closely with clinicians.
- Saw a kid with








