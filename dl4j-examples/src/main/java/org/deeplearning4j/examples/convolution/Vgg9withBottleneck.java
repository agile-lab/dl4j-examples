package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by paolo on 01/06/2017.
 */
public class Vgg9withBottleneck {

    public MultiLayerConfiguration conf(int width, int height, int iterations, int numLabels) {

        Map<Integer, Double> lrSchedule = new HashMap<>();

        lrSchedule.put(0, 0.002);
        lrSchedule.put(500, 0.001);
        lrSchedule.put(1000, 0.0005);
        lrSchedule.put(1500, 0.0003);
        lrSchedule.put(2000, 0.0001);

        MultiLayerConfiguration conf =

            new NeuralNetConfiguration.Builder()

                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.RELU)
                .updater(Updater.NESTEROVS)
                .activation(Activation.RELU)
                .learningRate(0.002)
                .regularization(true).l2(1e-4)
                .momentum(0.9)
                .list()

                // block 1

                .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nIn(3).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())

                .layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())

                .layer(2, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                // block 2

                .layer(3, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(32).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(32).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(5, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(128).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())

                .layer(6, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(32).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(32).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(8, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(128).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())

                .layer(9, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                // block 3

                .layer(10, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(12, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(1, 1).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())

                .layer(13, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(15, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())

                .layer(16, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(17, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
                .layer(18, new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(0, 0).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())

                .layer(19, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                // block 4

//                .layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//
//                    .padding(1, 1).nOut(512).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
//
//                .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//
//                    .padding(1, 1).nOut(512).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
//
//                .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//
//                    .padding(1, 1).nOut(512).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
//
//                .layer(13, new SubsamplingLayer.Builder()
//
//                    .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
//
//                    .stride(2, 2).build())

                // block 5

//                .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//
//                    .padding(1, 1).nOut(512).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
//
//                .layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//
//                    .padding(1, 1).nOut(512).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
//
//                .layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
//
//                    .padding(1, 1).nOut(512).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).build())
//
//                .layer(17, new SubsamplingLayer.Builder()
//
//                    .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
//
//                    .stride(2, 2).build())

                .layer(20, new DenseLayer.Builder().nOut(2048).dropOut(0.5).build())
                .layer(21, new DenseLayer.Builder().nOut(2048).dropOut(0.5).build())

                .layer(22, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(numLabels).activation(Activation.SOFTMAX).build())

                .backprop(true).pretrain(false).setInputType(InputType

                .convolutionalFlat(width, height, 3))

                .build();



        return conf;

    }

}
