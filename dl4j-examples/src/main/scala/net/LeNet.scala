package net

//import deeplogo.LogoClassificationS._
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by andlatel on 17/05/2017.
  */
class LeNet extends NetInterface with ConfigurationNet{

  override def createNet(): MultiLayerNetwork = {
    /**
      * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
      * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
      **/
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
      .regularization(false).l2(0.00001).activation(Activation.RELU).learningRate(0.0001)
      .weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.RMSPROP).momentum(0.9)
      .list
      .layer(0, convInit("cnn1", channels, 50, Array[Int](5, 5), Array[Int](1, 1), Array[Int](0, 0), 0))
      .layer(1, maxPool("maxpool1", Array[Int](2, 2)))
      .layer(2, conv5x5("cnn2", 100, Array[Int](5, 5), Array[Int](1, 1), 0))
      .layer(3, maxPool("maxool2", Array[Int](2, 2)))
      .layer(4, new DenseLayer.Builder().nOut(500).build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(numLabels).activation(Activation.SOFTMAX).build)
      .backprop(true).pretrain(false).setInputType(InputType.convolutional(height, width, channels)).build
    new MultiLayerNetwork(conf)
  }

}
