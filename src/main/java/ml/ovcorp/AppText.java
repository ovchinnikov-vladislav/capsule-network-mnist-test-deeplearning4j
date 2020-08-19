package ml.ovcorp;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.resources.Downloader;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.*;

public class AppText {

    private static final Logger log = LoggerFactory.getLogger(AppText.class);

    public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
    /**
     * Location to save and extract the training/testing data
     */
    public static final String DATA_PATH = FilenameUtils.concat("D:" + File.separator,
            "text_classification" + File.separator);

    public static final String WORD_VECTORS_PATH = new File("D:" + File.separator + "text_classification"+File.separator+"w2vec300" + File.separator,
            "GoogleNews-vectors-negative300.bin.gz").getAbsolutePath();

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");

        Random rng = new Random(12345);
        int batchSize = 5;
        int vectorSize = 300;               //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;                    //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

        NeuralNetwork nn;
        String type = "capsnet";
        switch (type) {
            case "cnn":
                 nn = buildCnn();
                break;
            case "capsnet":
                nn = CapsNet.buildForIMDB();
                break;
            default:
                nn = null;
        }

        log.info("Build CapsNet Model");

        nn.init();

        //Load word vectors and get the DataSetIterators for training and testing
        System.out.println("Loading word vectors and creating DataSetIterators");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        DataSetIterator trainIter = getDataSetIterator(true, wordVectors, batchSize, truncateReviewsToLength, rng);
        DataSetIterator testIter = getDataSetIterator(false, wordVectors, batchSize, truncateReviewsToLength, rng);

        if (nn instanceof ComputationGraph) {
            ComputationGraph net = (ComputationGraph) nn;

            System.out.println("Number of parameters by layer:");
            for(Layer l : net.getLayers() ){
                System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
            }

            log.info("Start training");
            net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
            net.fit(trainIter, nEpochs);


            //After training: load a single sentence and generate a prediction
            String pathFirstNegativeFile = FilenameUtils.concat(DATA_PATH, "aclImdb/test/neg/0_2.txt");
            String contentsFirstNegative = FileUtils.readFileToString(new File(pathFirstNegativeFile), (Charset) null);
            INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator)testIter).loadSingleSentence(contentsFirstNegative);

            INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);
            List<String> labels = testIter.getLabels();

            System.out.println("\n\nPredictions for first negative review:");
            for( int i=0; i<labels.size(); i++ ){
                System.out.println("P(" + labels.get(i) + ") = " + predictionsFirstNegative.getDouble(i));
            }

            net.save(new File("cnn.nn"));
        } else if (nn instanceof MultiLayerNetwork) {
            MultiLayerNetwork net = (MultiLayerNetwork) nn;

            System.out.println("Number of parameters by layer:");
            for(Layer l : net.getLayers() ){
                System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
            }


            log.info("Start training");
            net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
            net.fit(trainIter, nEpochs);

            net.save(new File("capsnet.nn"));
        }
    }

    public static MultiLayerNetwork buildCapsNet() {
        int vectorSize = 300;               //Size of the word vectors. 300 in the Google News model
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(12345)
                .updater(new Adam())
                .list()
                .setInputType(InputType.convolutional(truncateReviewsToLength, vectorSize, 1))
                .layer(new ConvolutionLayer.Builder()
                        .nOut(256)
                        .kernelSize(9, 9)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new PrimaryCapsules.Builder(8, 32)
                        .kernelSize(9, 9)
                        .stride(2, 2)
                        .build())
                .layer(new CapsuleLayer.Builder(3, 16, 3).build())
                .layer(new CapsuleStrengthLayer.Builder().build())
                .layer(new ActivationLayer.Builder(new ActivationSoftmax()).build())
                .layer(new LossLayer.Builder(new LossNegativeLogLikelihood()).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);

        return net;
    }

    public static ComputationGraph buildCnn() {
        int vectorSize = 300;               //Size of the word vectors. 300 in the Google News model
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        PoolingType globalPoolingType = PoolingType.MAX;

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.01))
                .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                //MergeVertex performs depth concatenation on activations: 3x[minibatch,100,length,300] to 1x[minibatch,300,length,300]
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                //Global pooling: pool over x/y locations (dimensions 2 and 3): Activations [minibatch,300,length,300] to [minibatch, 300]
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(globalPoolingType)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(2)    //2 classes: positive or negative
                        .build(), "globalPool")
                .setOutputs("out")
                //Input has shape [minibatch, channels=1, length=1 to 256, 300]
                .setInputTypes(InputType.convolutional(truncateReviewsToLength, vectorSize, 1))
                .build();

        ComputationGraph net = new ComputationGraph(config);
        return net;
    }

    public static void downloadDataImdb() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if (!directory.exists()) directory.mkdir();

        //Download file:
        String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "aclImdb";
        File extractedFile = new File(extractedPath);

        if (!archiveFile.exists()) {
            System.out.println("Starting data download (80MB)...");
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if (!extractedFile.exists()) {
                //Extract tar.gz file to output directory
                DataUtilities.extractTarGz(archizePath, DATA_PATH);
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    public static void checkDownloadW2VECModel() throws IOException {
        String defaultWordVectorsPath = FilenameUtils.concat("D:" + File.separator, "text_classification"+File.separator+"w2vec300");
        String md5w2vec = "1c892c4707a8a1a508b01a01735c0339";
        String wordVectorsPath = new File(defaultWordVectorsPath, "GoogleNews-vectors-negative300.bin.gz").getAbsolutePath();
        if (new File(wordVectorsPath).exists()) {
            System.out.println("\n\tGoogleNews-vectors-negative300.bin.gz file found at path: " + defaultWordVectorsPath);
            System.out.println("\tChecking md5 of existing file..");
            if (Downloader.checkMD5OfFile(md5w2vec, new File(wordVectorsPath))) {
                System.out.println("\tExisting file hash matches.");
                return;
            } else {
                System.out.println("\tExisting file hash doesn't match. Retrying download...");
            }
        } else {
            System.out.println("\n\tNo previous download of GoogleNews-vectors-negative300.bin.gz found at path: " + defaultWordVectorsPath);
        }
        System.out.println("\tWARNING: GoogleNews-vectors-negative300.bin.gz is a 1.5GB file.");
        System.out.println("\tPress \"ENTER\" to start a download of GoogleNews-vectors-negative300.bin.gz to " + defaultWordVectorsPath);
        Scanner scanner = new Scanner(System.in);
        scanner.nextLine();
        System.out.println("Starting model download (1.5GB!)...");
        Downloader.download("Word2Vec", new URL("https://dl4jdata.blob.core.windows.net/resources/wordvectors/GoogleNews-vectors-negative300.bin.gz"), new File(wordVectorsPath), md5w2vec, 5);
        System.out.println("Successfully downloaded word2vec model to " + wordVectorsPath);
    }

    private static DataSetIterator getDataSetIterator(boolean isTraining, WordVectors wordVectors, int minibatchSize,
                                                      int maxSentenceLength, Random rng) {
        String path = FilenameUtils.concat(DATA_PATH, (isTraining ? "aclImdb/train/" : "aclImdb/test/"));
        String positiveBaseDir = FilenameUtils.concat(path, "pos");
        String negativeBaseDir = FilenameUtils.concat(path, "neg");
        log.info("Load positive data from " + positiveBaseDir);
        log.info("Load negative data from " + negativeBaseDir);

        File filePositive = new File(positiveBaseDir);
        File fileNegative = new File(negativeBaseDir);

        Map<String, List<File>> reviewFilesMap = new HashMap<>();
        reviewFilesMap.put("Positive", Arrays.asList(Objects.requireNonNull(filePositive.listFiles())));
        reviewFilesMap.put("Negative", Arrays.asList(Objects.requireNonNull(fileNegative.listFiles())));

        LabeledSentenceProvider sentenceProvider = new FileLabeledSentenceProvider(reviewFilesMap, rng);

        return new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN2D)
                .sentenceProvider(sentenceProvider)
                .wordVectors(wordVectors)
                .minibatchSize(minibatchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .build();
    }

}
