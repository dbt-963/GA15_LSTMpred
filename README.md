
# GA15_LSTMpred
This model was constructed by deep learning and used amino acid descriptor as the characrization method. This model can predict the self-assembling ablity of hexpeptides.It is helpful to explore the sequence-aggregation relationship of peptides and easy to recognize the assemble-prone sequence in peptides and proteins

The GA15_LSTMpred package consists of two main modules: One is the data characterization module. Txt file of peptide sequences was used as input and descriptors were used to characterize peptides. Characteristic matrix was returned as Excel file. The second is the prediction module. The Excel document of descriptor matrix was used as input. GA15-LSTM model was used for prediction, and the prediction label and prediction score of peptide sequences were returned. Users can only use the data characterization module and obtain the descriptor Excel document for constructing and training their own model. Users can also import two modules at the same time to get the prediction results and scores directly.

## Requirements
In order to get started you will need:

Pytorch

Numpy

pandas

os
