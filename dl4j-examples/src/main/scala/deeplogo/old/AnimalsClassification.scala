package deeplogo.old

import java.io.File
import java.util
import java.util.Random

import org.apache.commons.io.FilenameUtils
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{FlipImageTransform, ImageTransform, WarpImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.distribution.{Distribution, GaussianDistribution, NormalDistribution}
import org.deeplearning4j.nn.conf.inputs.{InputType, InvalidInputTypeException}
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}


/**
  * Animal Classification
  *
  * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
  *
  * References:
  *  - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
  *  - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/leonyao_final.pdf
  *
  * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
  *  - Add additional images to the dataset
  *  - Apply more transforms to dataset
  *  - Increase epochs
  *  - Try different model configurations
  *  - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
  */
object AnimalsClassification {
  protected val log: Logger = LoggerFactory.getLogger(classOf[AnimalsClassification])
  protected var height = 100
  protected var width = 100
  protected var channels = 3
  protected var numExamples = 80
  protected var numLabels = 4
  protected var batchSize = 20
  protected var seed = 42
  protected var rng = new Random(seed)
  protected var listenerFreq = 1
  protected var iterations = 1
  protected var epochs = 20
  protected var splitTrainTest = 0.8
  protected var nCores = 2
  protected var save = false
  protected var modelType = "custom"

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    new AnimalsClassification().run(args)
  }
}

class AnimalsClassification {
  @throws[Exception]
  def run(args: Array[String]): Unit = {
    AnimalsClassification.log.info("Load data....")
    /** cd
      * Data Setup -> organize and limit data file paths:
      *  - mainPath = path to image files
      *  - fileSplit = define basic dataset split with limits on format
      *  - pathFilter = define additional file load filter to limit size and balance batch content
      * */
    val labelMaker = new ParentPathLabelGenerator
    val mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals/")
    val fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, AnimalsClassification.rng)
    val pathFilter = new BalancedPathFilter(AnimalsClassification.rng, labelMaker, AnimalsClassification.numExamples, AnimalsClassification.numLabels, AnimalsClassification.batchSize)
    /**
      * Data Setup -> train test split
      *  - inputSplit = define train and test split
      **/
    val inputSplit = fileSplit.sample(pathFilter, AnimalsClassification.numExamples * (1 + AnimalsClassification.splitTrainTest), AnimalsClassification.numExamples * (1 - AnimalsClassification.splitTrainTest))
    val trainData = inputSplit(0)
    val testData = inputSplit(1)
    /**
      * Data Setup -> transformation
      *  - Transform = how to tranform images and generate large dataset to train on
      **/
    val flipTransform1:ImageTransform = new FlipImageTransform(AnimalsClassification.rng)
    val flipTransform2:ImageTransform = new FlipImageTransform(new Random(123))
    val warpTransform:ImageTransform = new WarpImageTransform(AnimalsClassification.rng, 42)
    //        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
    val transforms: Seq[ImageTransform] = Seq(flipTransform1, warpTransform, flipTransform2)
    /**
      * Data Setup -> normalization
      *  - how to normalize images and generate large dataset to train on
      **/
    val scaler = new ImagePreProcessingScaler(0, 1)
    AnimalsClassification.log.info("Build model....")
    // Uncomment below to try AlexNet. Note change height and width to at least 100
    //        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

    val network = AnimalsClassification.modelType match {
      case "LeNet" => lenetModel
      case "AlexNet" => alexnetModel
      case "custom" => customModel
      case _ =>
        throw new InvalidInputTypeException("Incorrect model provided.")
    }
    network.init()
    val uiServer = UIServer.getInstance
    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    val statsStorage = new InMemoryStatsStorage //Alternative: new FileStatsStorage(File), for saving and loading later
    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)
    //Then add the StatsListener to collect this information from the network, as it trains
    val listeners = new util.ArrayList[IterationListener]
    listeners.add(new StatsListener(statsStorage))
    listeners.add(new ScoreIterationListener(AnimalsClassification.listenerFreq))
    network.setListeners(listeners)
    /**
      * Data Setup -> define how to load data into net:
      *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
      *  - dataIter = a generator that only loads one batch at a time into memory to save memory
      *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
      **/
    var recordReader = new ImageRecordReader(AnimalsClassification.height, AnimalsClassification.width, AnimalsClassification.channels, labelMaker)

    AnimalsClassification.log.info("Train model....")
    // Train without transformations
    recordReader.initialize(trainData, null)
    var dataIter = new RecordReaderDataSetIterator(recordReader, AnimalsClassification.batchSize, 1, AnimalsClassification.numLabels)
    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)
    var trainIter = new MultipleEpochsIterator(AnimalsClassification.epochs, dataIter, AnimalsClassification.nCores)
    network.fit(trainIter)
    // Train with transformations
    transforms.foreach(t => {
      System.out.print("\nTraining on transformation: " + t.getClass.toString + "\n\n")
      recordReader.initialize(trainData, t)
      dataIter = new RecordReaderDataSetIterator(recordReader, AnimalsClassification.batchSize, 1, AnimalsClassification.numLabels)
      scaler.fit(dataIter)
      dataIter.setPreProcessor(scaler)
      trainIter = new MultipleEpochsIterator(AnimalsClassification.epochs, dataIter, AnimalsClassification.nCores)
      network.fit(trainIter)
    })

    AnimalsClassification.log.info("Evaluate model....")
    recordReader.initialize(testData)
    dataIter = new RecordReaderDataSetIterator(recordReader, AnimalsClassification.batchSize, 1, AnimalsClassification.numLabels)
    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)
    val eval = network.evaluate(dataIter)
    AnimalsClassification.log.info(eval.stats(true))
    // Example on how to get predict results with trained model
    dataIter.reset()
    val testDataSet = dataIter.next
    val expectedResult = testDataSet.getLabelName(0)
    val predict = network.predict(testDataSet)
    val modelResult = predict.get(0)
    System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n")
    if (AnimalsClassification.save) {
      AnimalsClassification.log.info("Save model....")
      val basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/")
      ModelSerializer.writeModel(network, basePath + "model.bin", true)
    }
    AnimalsClassification.log.info("****************Example finished********************")
  }

  private def convInit(name: String, in: Int, out: Int, kernel: Array[Int], stride: Array[Int], pad: Array[Int], bias: Double) = new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build

  private def conv3x3(name: String, out: Int, bias: Double) = new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name(name).nOut(out).biasInit(bias).build

  private def conv5x5(name: String, out: Int, stride: Array[Int], pad: Array[Int], bias: Double) = new ConvolutionLayer.Builder(Array[Int](5, 5), stride, pad).name(name).nOut(out).biasInit(bias).build

  private def maxPool(name: String, kernel: Array[Int]) = new SubsamplingLayer.Builder(kernel, Array[Int](2, 2)).name(name).build

  private def fullyConnected(name: String, out: Int, bias: Double, dropOut: Double, dist: Distribution) = new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build

  def lenetModel: MultiLayerNetwork = {
    /**
      * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
      * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
      **/
    val conf = new NeuralNetConfiguration.Builder().seed(AnimalsClassification.seed)
      .iterations(AnimalsClassification.iterations).regularization(false)
      .l2(0.005)// tried 0.0001, 0.0005
      .activation(Activation.RELU).learningRate(0.0001).weightInit(WeightInit.XAVIER)  // tried 0.00001, 0.00005, 0.000001
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.RMSPROP).momentum(0.9).list.layer(0, convInit("cnn1", AnimalsClassification.channels, 50, Array[Int](5, 5), Array[Int](1, 1), Array[Int](0, 0), 0)).layer(1, maxPool("maxpool1", Array[Int](2, 2))).layer(2, conv5x5("cnn2", 100, Array[Int](5, 5), Array[Int](1, 1), 0)).layer(3, maxPool("maxool2", Array[Int](2, 2))).layer(4, new DenseLayer.Builder().nOut(500).build).layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(AnimalsClassification.numLabels).activation(Activation.SOFTMAX).build).backprop(true).pretrain(false).setInputType(InputType.convolutional(AnimalsClassification.height, AnimalsClassification.width, AnimalsClassification.channels)).build
    new MultiLayerNetwork(conf)
  }

  def alexnetModel: MultiLayerNetwork = {
    /**
      * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
      * and the imagenetExample code referenced.
      * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
      **/
    val nonZeroBias = 1
    val dropOut = 0.5
    val conf = new NeuralNetConfiguration.Builder().seed(AnimalsClassification.seed)
      .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.0, 0.01))
      .activation(Activation.RELU).updater(Updater.NESTEROVS)
      .iterations(AnimalsClassification.iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // normalize to prevent vanishing or exploding gradients
    .learningRate(1e-2).biasLearningRate(1e-2 * 2).learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.1).lrPolicySteps(100000).regularization(true).l2(5 * 1e-4).momentum(0.9).miniBatch(false).list.layer(0, convInit("cnn1", AnimalsClassification.channels, 96, Array[Int](11, 11), Array[Int](4, 4), Array[Int](3, 3), 0)).layer(1, new LocalResponseNormalization.Builder().name("lrn1").build).layer(2, maxPool("maxpool1", Array[Int](3, 3))).layer(3, conv5x5("cnn2", 256, Array[Int](1, 1), Array[Int](2, 2), nonZeroBias)).layer(4, new LocalResponseNormalization.Builder().name("lrn2").build).layer(5, maxPool("maxpool2", Array[Int](3, 3))).layer(6, conv3x3("cnn3", 384, 0)).layer(7, conv3x3("cnn4", 384, nonZeroBias)).layer(8, conv3x3("cnn5", 256, nonZeroBias)).layer(9, maxPool("maxpool3", Array[Int](3, 3))).layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005))).layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005))).layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(AnimalsClassification.numLabels).activation(Activation.SOFTMAX).build).backprop(true).pretrain(false).setInputType(InputType.convolutional(AnimalsClassification.height, AnimalsClassification.width, AnimalsClassification.channels)).build
    new MultiLayerNetwork(conf)
  }

  def customModel: MultiLayerNetwork = {
    /**
      * Use this method to build your own custom model.
      **/
    null
  }
}
