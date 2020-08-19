package ml.ovcorp;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;

public class CapsNet {

    private CapsNet() {}

    public static MultiLayerNetwork buildForMNIST() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(123)
                .updater(new Adam())
                .list()
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
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
                .layer(new CapsuleLayer.Builder(10, 16, 1).build())
                .layer(new CapsuleStrengthLayer.Builder().build())
                .layer(new ActivationLayer.Builder(new ActivationSoftmax()).build())
                .layer(new LossLayer.Builder(new LossNegativeLogLikelihood()).build())
                .build();

        return new MultiLayerNetwork(conf);
    }

    public static ComputationGraph buildForIMDB() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(123)
                .updater(new Adam())
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(64, 64, 1))
                .addLayer("cnn", new ConvolutionLayer.Builder()
                        .nOut(256)
                        .kernelSize(9, 9)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build(), "input")
                .addLayer("primary_capsules", new PrimaryCapsules.Builder(8, 32)
                        .kernelSize(9, 9)
                        .stride(2, 2)
                        .build(), "cnn")
                .addLayer("digit_capsules", new CapsuleLayer.Builder(3, 16, 1).build(), "primary_capsules")
                .addLayer("decoder1", new CapsuleStrengthLayer.Builder().build(), "digit_capsules")
                .addLayer("decoder2", new ActivationLayer.Builder(new ActivationSoftmax()).build(), "decoder1")
                .addLayer("decoder3", new LossLayer.Builder(new LossNegativeLogLikelihood()).build(), "decoder2")
                .setOutputs("decoder3")
                .build();

        return new ComputationGraph(conf);
    }
}
