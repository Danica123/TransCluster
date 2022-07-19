# TransCluster
Recent advances in single-cell RNA sequencing (scRNA-seq) have accelerated the development of techniques to classify thousands of cells through transcriptome profiling. As more and more scRNA-seq data became available, supervised cell type classification methods using externally well-annotated source data became more popular than unsupervised clustering algorithms. However, accurate cellular annotation of single cell transcription data remains a significant chal-lenge. Here, we propose a hybrid network structure called TransCluster, which uses linear discri-minant analysis and multi-headed self-attention mechanisms to enhance feature learning. It is a cell-type identification tool for single-cell transcriptomic maps. It shows high accuracy and ro-bustness in many cell data sets of different human tissues. It is superior to other known methods in external test data set. To our knowledge, TransCluster is the first attempt to use Transformer for annotating cell types of scRNA-seq, which greatly improves the accuracy of cell-type annotation.
# Install
* Python 3.6.13
* keras 2.4.3
* numpy 1.19.5
* pandas 1.1.5
* scikit-learn 0.24.2
* scipy 1.4.1
* tensorflow 2.4.1
# Usage(Take Spleen for example)
* First, download the original data from the Releases-Dataset, such as human_Spleen9887_celltype.csv and human_Spleen9887_data.csv, and divide it into training set and test set.
* Next, use the lda.py file to perform dimensionality reduction of the original data.
* Then, the Boostmodel-CNN_LDA_Transformer_spleen_save.py file is used to perform the prediction of cell categories, generating a file that contains the likelihood of each category.
* Finally, the performance of the method is evaluated with testACC_spleen.py and the results are visualized with files in VisualAnalysis.
# Data availability
The data used in the method are stored in the Releases-Dataset.
