# 21st July 2025
001 - Vari-Cozi - Alexander Sasse @ 11:20
- most varient lie in non-coding region
- You can do Machenis Insights and Varient effect prediction
- Problems are with models: (Emformer - S2F understand promotor but not enhancer)
- Constraining training on varients improves the prediction of differences.
- Deep-Allele improves for allelic ratios. 
- SAGE-net trains on personal genomes. ( P-SAGE cannot improve predictions for unseen genes)
- Training on sequence varinet is promissing.


002 - Combining massively parallel reporter assays and graph genomics to assay the regulatory effects of indels and structural variants
- SuRE ()
- They used the Graph Genome. 
- SuRE-SV pipeline in Nextflow.
- 

003 - Multilingual model improves zero-shot prediction of disease effects on proteins @ 11:21
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


005 - Functional characterisation of standing variation around disease-associated genes using Massively Parallel Reporter Assays - Killian
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
- Project, Development of Digital Twons for Rare Disease.
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
- Saw a kid with some problem but with the blood sample no infomration.
- takes seq as input and predicts splicing prob in 15 tissues.
- Benchmarked and supoeior performance from SPlicaeAI.
- Tissue apecificipic splice alteration compariaon with expreasion.
- want to seprate splicing mechanisms worth expression mechanisms.
- identified novel genes that have never been reported.
- better understand the relationship with a network view.
- took a chhort of 95 patients and used that sequencing data for evaluation.
- benchmarked on pre-trained gLMs for RNA prediction.


# 010 - Slivka is a Python Client | You might want web portal for a better publication
- Web, API client
- Dundee Resourse (collection of many websurver)
- JalView
- Conventional web App rather than this they already have the configuration
- Locally installed Command Line tools.
- 

# 011 Keynote EASYstrata: a fully integrated workflow to infer evolutionary strata along sex chromosomes and other supergenes
-

# 012 Quantum Approximate Optimization for K-Area Clustering of Biological Data


# 22nd July 2025
# 001 Keynote Computational biology in the age of AI agents
- They have created different agents, PI, Immunologist and so on.
- They can have parallel meetings and so on.
-  PI agents read the transcripts and suggest what the next steps could be.
-  It is a six-step process.
-  Virtual Lab Designs nanobodies for recent COVID variants.
-  It created a framework for designing nanobodies.
-  Agents generate Jupyter Notebooks and reports.
-  CellVoyager - checks data and reports and generates a new report.
-  They validated the report by the original authors.
-  Case Study1: COVID19 response, data exploration and visualization, CD8 T cells in COVID
-  Case Study2: Agging in Brain: Predict biological age, agent found strong coorelation between transcriptional noise and aging.
- Agents can help us understand AI models usually blackbox models.
- They validated the LLM generated SAE description.
- Limitations:
  - Better at using exisisting tools then creating new tools. they can guide which tools to use.
  - More breath than depth.
  - 


# 002 Toward Mechanistic Genomics: Advances in Sequence-to-Function Modeling
Recent advances have firmly established sequence-to-function models as essential tools in modern genomics, enabling unprecedented insights into how genomic sequences drive molecular and cellular phenotypes. As these models have matured—with increasingly robust architectures, improved training strategies, and the emergence of standardized software frameworks—the field has rapidly evolved from proof-of-concept demonstrations to widespread practical applications across a variety of biological systems.

With the core methodologies now widely adopted and infrastructure in place, the community's focus is shifting toward ambitious new frontiers. There is growing momentum around developing models that are biologically interpretable, capable of uncovering causal mechanisms of gene regulation, and generalizable to novel contexts—such as predicting the effects of perturbing a regulatory protein rather than simply altering a DNA sequence. These efforts reflect a broader aspiration: to create models that serve not just as black-box predictors, but as scientific instruments that deepen our understanding of genome function.

In this talk, we will explore how such models can move us from descriptive genomics to mechanistic insight, highlighting recent innovations in architecture and training that support interpretability, modularity, and reusability. We will examine the contexts in which these models offer clear advantages, the limitations that remain, and practical considerations for their training. Ultimately, we will consider how advancing these models may refine the role of machine learning in biology, supporting not only accurate prediction but also the generation of more detailed and mechanistically informed hypotheses.
- Puffin Model
  - PromotorAl new model for promotor sequence for varient interpertation.
  - PuffinD is the deeplearning model and puffin is not even there.
  - Puffin is doing really bad on rare varients.
  - no- trineucleotides effects in the puffin-notri and there was incredible improvement.
  - Interpretable models may not out-perform blackbox models.
- working on long range interactions.
  -  Loop extrusion model for 3D organisation.
  -  Created differentialable loop exclusion model dLEM.
  -  We can fit diverse patterns and capture cell-type differences.
  -  What controls cohesin velocity?
    - CTCF controls it. CTCF features were made directional.
    - they had CTCF+ and CTCF=
    - WAPL controls the cohesion  unloading.
    - The difference in WAPL and Un-Treated was based on lambda.
- Conclusion
  - Interpretable models will might not be able to ever outperform the blackbox models.
  - 

  # 003 Deep learning models for unbiased sequence-based PPI prediction plateau at an accuracy of 0.65
- Repoted accuracies of protein-protein interaction preditcion (PPI) we really well.
- Predicting the interaction from sequence alone is a hard problem.
- So models must be able ot use some kind of shortcut.
- They constructed gold-standard dataset. (No protein Overlap b/w training and test set)
- They were getting 0.065 accuracy by the ESM-2 embedding.
- There is a sweet spot in the embedding size.
- Embedding can be helpful in prediction, and could be a shortcut.
- Models profit from ESM embeddings.


# 004 - Proceedings Presentation: Accurate PROTAC targeted degradation prediction with DegradeMaster




# 005 Func - Cosi = GOAnnotator: Accurate protein function annotation using automatically retrieved literature

- Automated protein function prediction/annotation (AFP) is vital for understanding biological processes and advancing biomedical research. Existing text-based AFP methods including the state-of-the-art method, GORetriever, rely on expert-curated relevant literature, which is costly and time-consuming, and covers only a small portion of the proteins in UniProt. To overcome this limitation, we propose GOAnnotator, a novel framework for automated protein function annotation. It consists of two key modules: PubRetriever, a hybrid system for retrieving and re-ranking relevant literature, and GORetriever+, an enhanced module for identifying Gene Ontology (GO) terms from the retrieved texts. Extensive experiments over three benchmark datasets demonstrate that GOAnnotator delivers high-quality functional annotations, surpassing GORetriever by uncovering unique literature and predicting additional functions. These results highlight its great potential to streamline and enhance the annotation of protein functions without relying on manual curation.
- Funn information retrival and then fine grained retrival.
- Documents are collected and then are prioritize
- Not all the literature is relavent to the target.


# 006 Semi-Supervised Data-Integrated Feature Importance Enhances Performance and Interpretability of Biological Classification Tasks

- Accurate model performance on training data does not ensure alignment between the model’s feature weighting patterns and human knowledge, which can limit the model’s relevance and applicability. We propose Semi-Supervised Data-Integrated Feature Importance (DIFI), a method that numerically integrates a priori knowledge, represented as a sparse knowledge map, into the model’s feature weighting. By incorporating the similarity between the knowledge map and the feature map into a loss function, DIFI causes the model’s feature weighting to correlate with the knowledge. We show that DIFI can improve the performance of neural networks using two biological tasks. In the first task, cancer type prediction from gene expression profiles was guided by identities of cancer type-specific biomarkers. In the second task, enzyme/non-enzyme classification from protein sequences was guided by the locations of the catalytic residues. In both tasks, DIFI leads to improved performance and feature weighting that is interpretable. DIFI is a novel method for injecting knowledge to achieve model alignment and interpretability.
- Semi-supervised feature importance enhances elasification performance.
- They want to use the embeddings from pre-trained models.


# 007 | On the completeness, coherence, and consistency of protein function prediction: lifting function prediction from isolated proteins to biological systems
- Types of Coherence Dependance.
- Computational methoda do not produce biological plausible annotations.


# 008 
Unravelling the pangenome of autotrophic bacteria: Metabolic commonalities, evolutionary relationships, and industrially relevant traits
- HydroCow
- Generating milk from bacteria
- There is global carbon emmision (moving to sustanable things)
- Food sector contributes 26% to carbon emmision.
- SOLEIN: already commercialized in Singapore.
- We need to understand the metabolism of the bacteria for effectie energy transfer.
- Generate pangenomes (Autotrophic Pangenomes)
- They found genes in energy transfer.
- 




# 009 Fair | RO-Crate: Capturing FAIR research outputs in bioinformatics and beyond
- Packaging data with the metadata.
- Capturing the how, what and where of your research.
- Used by
  - Elixir (BioDT, Galaxy)
  - Data Plant
  -  Biodiversity Genomics Europe (COPO)
  -  Can be used in workflows, Data Archiving and TRE's (Nextflow, RoHub)


# 010 PheBee | A Graph-Based System for Scalable, Traceable, and Semantically Aware Phenotyping
- Phenotype used in cohort building
- fits in better in the hospital infrastructure
- A graph is made and in the middle we can see the graph by zooming out.
- Translational Use Cases (NeoGx, CAVAlio)




# 011 Disentangling the genetic and non-genetic origin of disease co-occurrences
Numerous diseases co-occur more than expected by chance, likely due to a combination of genetic and environmental factors. However, the extent to which these influences shape disease relationships remains unclear. Here, we integrate large-scale RNA-seq data and heritability measures from human diseases with genomic data from the UK Biobank to disentangle the genetic and non-genetic origins of disease co-occurrences (DCs).
Our findings show that gene expression not only recovers but also expands upon genomically explained DCs, capturing disease relationships beyond genetic variation. Approximately 60% of transcriptomically inferred DCs have a detectable genomic component, whereas the remaining 40% are not explained by known genomic layers, suggesting contributions from regulatory or environmental mechanisms. Consistent with this interpretation, the relative contributions of transcriptomics and genomics reconstruct disease etiology and correlate with comorbidity burden, revealing key aspects of disease mechanisms. Complex diseases with strong genetic predispositions tend to be captured by both omics, whereas those primarily influenced by non-genetic factors are better explained by transcriptomics. Additionally, we find that diseases do not generally co-occur based on their heritability, except when sharing SNPs. However, highly heritable diseases tend to have genetically driven co-occurrences, even with lowly heritable diseases. In contrast, transcriptomics explains DCs regardless of heritability, at least partly due to non-heritable mechanisms, such as regulatory or environmental. Integrating transcriptomic and genomic data provides near-complete coverage of DCs among the analyzed diseases, with a considerable portion likely rooted in factors beyond DNA sequence and, therefore, potentially modifiable.
- Dependancey between molecular layers
- How do disease distribute in molecular space?
  - Most diseases are well captured by transcriptomics.
  - Type-1 diabeties are better captured with genomics.
  -   60% of transcriptomically infered DC's are supported by components.
  -   Integerating transcriptomic and genomic data provides near complete picture. 



# 23 July 2925
# 001  Keynote Charlotte Deane @ 9am
- Paragraph Tool
  - you give seq of antibody.
  - probability of each residue to bind to the target.
- AbLang
  - A language model for antibodies


# 002 Learning variant effects on chromatin accessibility and 3D structure without matched Hi-C data @ 11:20
- Valentina Boeva,ETH Zurich, Switzerland
Chromatin interactions provide insights into which DNA regulatory elements connect with specific genes, informing the activation or repression of gene expression. Understanding these interactions is crucial for assessing the role of non-coding mutations or changes in chromatin organization due to cell differentiation or disease. Hi-C and single-cell Hi-C experiments can reveal chromatin interactions, but these methods are costly and labor-intensive.
Here, I will introduce our computational approach, UniversalEPI, an attention-based deep ensemble model that predicts regulatory interactions in unseen cell types with a receptive field of 2 million nucleotides, relying solely on DNA sequence data and chromatin accessibility profiles. Demonstrating significantly better performance than state-of-the-art methods, UniversalEPI—with a much lighter architecture—effectively predicts chromatin interactions across malignant and non-malignant cancer cell lines (Spearman’s Rho > 0.9 on unseen cell types).
To further expand its applicability, we integrate ASAP, our deep learning toolset that predicts the effects of genomic variants on ATAC-seq profiles. These predicted accessibility profiles can serve as input to UniversalEPI. Importantly, the accuracy of Hi-C interaction prediction remains virtually unchanged when replacing experimental ATAC-seq profiles with those generated by ASAP, indicating strong robustness and enabling predictions even in the absence of experimental accessibility data.
This combined framework represents an advancement in in-silico 3D chromatin modeling, essential for exploring genetic variant impacts on disease and monitoring chromatin architecture changes during development.
- We know dna accacibilyt and we try to model the dna folding.
- They use 2 models,
- if the gate is open the information will pass if the gate is closed the information will not pass.
- it also predicts about the uncertinity.



003 HitSeq Spatial transcriptomics deconvolution methods generalize well to spatial chromatin accessibility data
- The problem is the that there could be multiple cells in one spot.
- so you get agrigated signal.
- Tries to learn the mappings from the reference dataset to the cell data.
- The performance decreases with larger reference datasets.
- 



004 The practical use of AI in cores
- three groups ChatGPT, mets code LLM, just search engine (use carefully)
- Sequera
  -  is company behind nextflow, squera could be free for academic program.
  - SqueraAI for bioinformatic LLM for nextflow. (could also be integerated into VS code)
  - MULTIQC (Unified reporting across platform runs) and AI chat.
- Madeline Gogol: 
  - Benchmarking is a problem there.
  - Finetuning to do particular tasks.
- Training course
  - Training internally and externally.
  - Enabeling not replacing people.
- Copahegan university said you are only allowed to use copilot.

005 Identification and characterisation of chromatin-associated long non-coding RNAs in human
- Chromatin-associated long non-coding RNAs (ca-lncRNAs) play crucial regulatory roles within the nucleus by preferentially binding to chromatin. Despite their importance, systematic identification and functional studies of ca-lncRNAs have been limited. Here, we identified and characterized human ca-lncRNAs genome-wide, utilising 323,950 lncRNAs from LncBook 2.0 and integrating high-throughput sequencing datasets that assess RNA-chromatin association. We identified 14,138 high-confidence ca-lncRNAs enriched on chromatin across six cell lines, comprising nearly 80% of analyzed chromatin-associated RNAs, highlighting their significant role in chromatin localisation. To explore the sequence basis for chromatin localisation, we applied the LightGBM machine learning model to identify contributing nucleotide k-mers and derived 12 sequence elements through k-mer assembly and feature ablation. These sequence elements are frequently found within Alu repeats, with more Alu repeats enhancing chromatin localisation. Meta-profiling of chromatin-binding sequencing segments further demonstrated that ca-lncRNAs bind to chromatin through Alu repeats. To delve deeper into the molecular mechanisms underlying the binding, we conducted integrative interactome analysis and computational prediction, revealing that Alu repeats primarily tether to chromatin through dsDNA-RNA triplex formation. Finally, to address sample constraints in ca-lncRNA identification, we developed a machine learning model based on sequential feature selection for large-scale prediction. This approach yielded 201,959 predicted ca-lncRNAs, approximately 70% of which are predicted to be preferentially located in the nucleus. Collectively, these high-throughput-identified and machine-learning-predicted ca-lncRNAs together form a robust resource for further functional studies.
- Ln Rna longer than 200 neucleotides
- lnRNA play crutial role in the nucleus,
- 60% lncRNA are in chromatin
- RNA chromatin interaction sequencing data.
- are of the CLE's are located in Alu
- 50 lncRNA contain Ali element.
- more than 30% lncRNAs contain multiple Alu.
- Alu is Direct Chromatin Binding Region in Ca-lncRNA.
- Alu boundries exibit higher chromatin binding density.
- in NEAT1 there are three peaks in the chromatin binding.
- There are 4 hypothesis for this but they do not know which one is the major mechanism
- Alu regions exibit higher propancity to form triplex structures.



006 - Benchmarking Variant-Calling Workflows: The nf-core/variantbenchmarking Pipeline within the GHGA Framework
- nf-core/varientbenchmarking pipeline in GHGA
- GHGA is not just an archive
- THis big database is gonna have alnalysis.
- At the end of benchmarking they will share the reslts.
- comparing the snv is easy.
- Structural varints it is defficult as the representations vary a lot.
- Varient representation and normalization is different.
- Genomic complexity.
- No standardization.
- No best practice to do SV bench marking.
- Pipeline used GS test set GIAB.
- Liftover is needed when it is aligned to a different reference.
- Reformating and standardizing the SV representsation.
- Varien filtering depends on the length.
- 


007 - SpliSync: Genomic language model-driven splice site correction of long RNA reads | Liliana Florea - HiTSeq: High Through

- We developed SpliSync, a deep learning method for accurate splice site correction in long read alignments. It combines a genomic language model, HyenaDNA, and a 1D U-net segmentation head, integrating genome sequence and alignment embeddings. SpliSync improves the detection of splice sites and introns and, when integrated with a short read transcript assembler, allows for improved transcript reconstruction, matching or outperforming reference methods like IsoQuant and FLAIR. The method shows promise for transcriptomic applications, especially in species with incomplete gene annotations or for discovering novel transcript variations.

- performed evaluation on simulated data as well as real data (Human WTC11).
- she think that she has showen that it is more accurate then the previous methods.



# 008 adverSCarial: a toolkit for exposing classifier vulnerabilities in single-cell transcriptomics
- Adversarial attacks pose a significant risk to machine learning (ML) tools designed for classifying single-cell RNA-sequencing (scRNA-seq) data, with potential implications for biomedical research and future clinical applications. We present adverSCarial, a novel R package that evaluates the vulnerability of scRNA-seq classifiers to various adversarial perturbations, ranging from barely detectable, subtle changes in gene expression to large-scale modifications. We demonstrate how five representative classifiers spanning marker-based, hierarchical, support vector machine, random forest, and neural network algorithms, respond to these attacks on four hallmarks scRNA-seq datasets. Our findings reveal that all classifiers eventually fail under different amplitudes of perturbations, which depend on the ML algorithm they are based on and on the nature of the modifications.
Beyond security concerns, adversarial attacks help uncover the inner decision-making mechanisms of the classifiers. The various attack modes and customizable parameters proposed in adverSCarial are useful to identify which gene or set of genes is crucial for correct classification and to highlight the genes that can be substantially altered without detection. These functionalities are critical for the development of more robust and interpretable models, a step toward integrating scRNA-seq classifiers into routine research and clinical workflows. The R package is freely available on Bioconductor (10.18129/B9.bioc.adverSCarial) and helps evaluate scRNA-seq-based ML models vulnerabilities in a computationally-cheap and time-efficient framework.

- Single Gene (4/5 classifiers vernerable)
- Max-Change Method.
- CGD Method.
- When we put very abarent values for the modification.

  
# 009 Quality assessment of long read data in multisample lrRNA-seq experiments using SQANTI-reads
- Groups reads into unique junction chains.
  - 
- Identified putatuve unabbotated transcripts.
- Quantifies variations around annotated junctions.
- Multi-Sample visualization and PCA


# 010 longread pacbio
- can we process 100M reqds in a few hours
- develope a scalable caller 
- how much a very high through put pacbio cost

# 011 Quantifying rna expressions and modifications using long reqd RNA-seq
- Jonathan Görke - m6anet
- talked a lot about singapore
- pandemic ended because of mRNA vaccines
- talked about introducing RNA
- strong inflamatory response but protein was nit produced
- they replaced unmodified bases with modified bases 
- we can also use un-natural modifications
- m6a modification have an impact
- direct RNA sequencing
- we can also get signals for modifications 
- identifying the modifications is largely a computational task
- some of those rna are modified
- dorado used for ont basecaller
- can we train modification models for other models?
- framework is generalizable
- these models are trained on data that is very limited 
- kmers having more then one modifications model does not have seen 
- so it causes the problems 
- we can see typical emrichment on 3prime end 
- there could be a large number of false positives



# 25th July 2025
# Nano Pore HitSeq | Resolving Paralogues and Multi-Copy Genes with Nanopore Long-Read Sequencing | Sergey Nurk
- Used CTC-Idea: you have a path and you ask what is used in the base caller.
- Dupelx Sequencing: enables Q30 accuracy.
- Repeated measurements reduce the errors.
- Stereo Duplex:
  - previous approah pair decoding, 2021 is relatively ancient technique.
  - decode through both but it is computationaly expansive 700mb per second.
  - Used Heuristic algorithm, fast and transparent high speed.
  - Streo is best of both worlds
  - basecaller needs to be robust.
- ModCaller training that DNA can be modified.


# GenCompBio | Haplotype-specific copy number profiling of cancer genomes from long reads sequencing data
- Savana, Purpe for coppy number profiling.
- WaKhan, haplotype specific copy number caller.
- It generated BED files, VCF files as well as interactive plots.
- No benchmark Truth set avalible for copy numbers.
- All allele specific tools will strugle because they will assign.
- Already integerated in PacBio Workflow.
- can you tell if most of the improvement came by using long read data, he thinks it is because of phase correction.
- for shortneed we need phased VCF to work.


# HitSeq | Exploiting uniqueness: seed-chain-extend alignment on elastic founder graphs
- they simply construction by representing aligned sequences.
- N50 matric is used to check the composition of the dna/chromosome
- Only require few basic operations on the text.
- benefits of chaining is that
- They do not handle large MSA graphs.
- Didn't looked at meaning full biology.

# HitSeq | FroM Superstring to Indexing: a space-efficient index for unconstrained k-mer sets using the Masked Burrows-Wheeler Transform (MBWT)
- we want dictionary queries
- They use de Brujin graphs. 


# HitSeq | The Alice assembler: dramatically accelerating genome assembly with MSR sketching | Roland Faure
- Meta Genome assembly | Assembling gut Metagenome (HiFi 250Gb)
- metaMDGB is extremly fast, 19h 10G RAM
- It used minimizers, is a nice trick.
- It is fast but varients are lost.
- Mapping friendly sequencing reductions.
- rather than assembling the orignal reads you assemble the sketch
- Alice assembler is becoming kind of stable now.
- 5h and 10gb of RAM.
- DARK side: it also amplifies the sequencing erros too.
- Questions: Did you test ths on short read data and how does that come out? It works well with long reads.
  - Q: How do you choose the hash function? as there are different kinds of moss in MSR did you try different hash functions to get the best one? used cahotic hash function NT-hash.
  - large genomes can also be compressed using sketching.
    

# HitSeq | BINSEQ: A Family of High-Performance Binary Formats for Nucleotide Sequences

- Sequencing data is getting bigger very fast.
- FASTQ; is probalamatic with paralization
- Records has to be parsed sequently
- How to improve FASTQ?
  - family of binary format BQ and VBQ.
  - BQ: Simple format, does not have quality info
  - VBQ: Optional quality info, varient record possible   



# HitSeq | Minimizers | GreedyMini: Generating low-density DNA minimizers | Arseny Shur, Bar Ilan University, Israel


# HitSeq | LYCEUM: Learning to call copy number variants on low coverage ancient genomes
- Ancient Genome analysis is limited to SNV's
- Why Chellanging
  - High COntamination
  - Chemical env Demage
  - Low Coverae
  - ECOLE: Learnign to call the copy number variatios.
  - Train model on mdern data and then fine-tune it to the ancient dna conditions.
- They masked the positons and trained the model to get back those things to get better copy number calls.


# RegSys | Predicting gene expression using millions of yeast promoters reveals cis-regulatory logic

- Randomn Promotor Dream Chellange.
- We have promotor sequences and expression values.
- More improvement on the challenging sequences and low on SNV.
- Why we should care about this?
  -  Complex cis-regulatory logic can be discused.
https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf130/8155182


# RegSys | Learning the Regulatory Genome by Destruction and Creation
- Crisper_SURF
- Crisper-Clear
  - Trying to extablish genotype to phenotype relationship.
  - Established framework to increase resolution of CRISPR tiling screen.
  - Clear Screen:
    - You need to decide the editor. (base Editing C-T and A-G)
    - Region

   













