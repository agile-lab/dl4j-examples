package net
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
class CustomNet extends NetInterface with ConfigurationNet{

  override def createNet(): MultiLayerNetwork = {
    val nonZeroBias: Double = 1
    val dropOut: Double = 0.5
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.0, 0.01))
      .activation(Activation.RELU)
      .updater(Updater.NESTEROVS)
      .iterations(iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(0.0001).biasLearningRate(0.001 * 2)
      .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.1).lrPolicySteps(100000)
      .regularization(true).l2(5 * 1e-4).momentum(0.70)//.miniBatch(false)
      .list
      .layer(0, new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name("cnn1").nIn(channels).nOut(32).biasInit(nonZeroBias).activation(Activation.RELU).build)
      .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).name("maxpool1").build )
      .layer(2, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1), Array[Int](2, 2)).name("cnn3").nOut(64).biasInit(0).activation(Activation.RELU).build)
      .layer(3, new BatchNormalization.Builder().name("btcNorm").build)
      .layer(4, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1), Array[Int](2, 2)).name("cnn4").nOut(64).biasInit(0).activation(Activation.RELU).build)
      .layer(5, new SubsamplingLayer.Builder(PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).name("maxpool1").build )
      .layer(6, fullyConnected("ffn1", 256, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
      .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(numLabels).activation(Activation.SOFTMAX).build)
      .backprop(true)
      .pretrain(false)
      .setInputType(InputType.convolutional(height, width, channels))
      .build
    new MultiLayerNetwork(conf)
  }
}
