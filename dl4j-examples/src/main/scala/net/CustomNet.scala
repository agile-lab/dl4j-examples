package net
import deeplogo.Configuration
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.distribution.{GaussianDistribution, NormalDistribution}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by andlatel on 17/05/2017.
  */
class CustomNet(conf: Configuration) extends NetInterface{

  override def createNet(): MultiLayerNetwork = {
    val nonZeroBias: Double = 1
    val dropOut: Double = 0.5
    val networkConf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(conf.seed)
      .weightInit(WeightInit.RELU).dist(new NormalDistribution(0.0, 0.01))
      .activation(Activation.RELU)
      .updater(Updater.NESTEROVS)
      .iterations(conf.iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(0.0008).biasLearningRate(0.00001 * 2)
      .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.0002).lrPolicySteps(10000)
      .regularization(true).l2(5 * 1e-4).momentum(0.9).miniBatch(true)
      .list
      .layer(0, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](2, 2), Array[Int](1, 1)).name("cnn1").nIn(conf.channels).nOut(128).biasInit(nonZeroBias).activation(Activation.RELU).build)
      .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, Array[Int](2, 2), Array[Int](1, 1), Array[Int](0, 0)).name("maxpool1").build )
      .layer(2, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](2, 2), Array[Int](1, 1)).name("cnn3").nOut(64).biasInit(0).activation(Activation.RELU).build)
      .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX, Array[Int](2, 2), Array[Int](1, 1), Array[Int](0, 0)).name("maxpool2").build )
      //.layer(4, new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name("cnn4").nOut(64).biasInit(0).activation(Activation.RELU).build)
      //.layer(5, new SubsamplingLayer.Builder(PoolingType.AVG, Array[Int](2, 2), Array[Int](1, 1), Array[Int](0, 0)).name("avgpool3").build )
      //.layer(5, new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name("cnn5").nOut(64).biasInit(0).activation(Activation.RELU).build)
      .layer(4, new DenseLayer.Builder().name("ffn1").nOut(256).biasInit(0).dropOut(dropOut).dist(new GaussianDistribution(0, 0.01)).build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(conf.numLabels).activation(Activation.SOFTMAX).build)
      .backprop(true)
      .pretrain(false)
      .setInputType(InputType.convolutional(conf.height, conf.width, conf.channels))
      .build
    new MultiLayerNetwork(networkConf)

    /*** Result
      *
      */

    /*override def createNet(): MultiLayerNetwork = {
      val nonZeroBias: Double = 1
      val dropOut: Double = 0.5
      val networkConf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
        .seed(conf.seed)
        .weightInit(WeightInit.RELU).dist(new NormalDistribution(0.0, 0.01))
        .activation(Activation.RELU)
        .updater(Updater.NESTEROVS)
        .iterations(conf.iterations)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(0.01).biasLearningRate(0.001 * 2)
        .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.002).lrPolicySteps(100)
        .regularization(true).l2(5 * 1e-2).momentum(0.85).miniBatch(true)
        .list
        .layer(0, new ConvolutionLayer.Builder(Array[Int](7, 7), Array[Int](3, 3), Array[Int](1, 1)).name("cnn1").nIn(conf.channels).nOut(128).biasInit(nonZeroBias).activation(Activation.RELU).build)
        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, Array[Int](2, 2), Array[Int](2, 2), Array[Int](0, 0)).name("maxpool1").build )
        .layer(2, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1), Array[Int](1, 1)).name("cnn3").nOut(32).biasInit(0).activation(Activation.RELU).build)
        //.layer(3, new BatchNormalization.Builder().name("btcNorm").build)
        //.layer(3, new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name("cnn4").nOut(16).biasInit(0).activation(Activation.RELU).build)
        .layer(3, new SubsamplingLayer.Builder(PoolingType.AVG, Array[Int](2, 2), Array[Int](1, 1), Array[Int](0, 0)).name("maxpool1").build )
        //.layer(7, new SubsamplingLayer.Builder(PoolingType.AVG, Array[Int](2, 2), Array[Int](1, 1), Array[Int](0, 0)).name("maxpool1").build )
        .layer(4, conf.fullyConnected("ffn1", 256, nonZeroBias, dropOut, new GaussianDistribution(0, 0.01)))
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(conf.numLabels).activation(Activation.SOFTMAX).build)
        .backprop(true)
        .pretrain(false)
        .setInputType(InputType.convolutional(conf.height, conf.width, conf.channels))
        .build
      new MultiLayerNetwork(networkConf)*/
  }


}
