package deeplogo

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import annotation.{AnnotatedImageReader, AnnotationDataSet, SelectiveSearchAnnotation}
import deeplogo.old.AnimalsClassification
import deeplogo.old.LogoClassification.{height, log, width}
import org.apache.commons.io.FilenameUtils
import org.datavec.api.io.filters.{BalancedPathFilter, RandomPathFilter}
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.{FileSplit, InputSplit}
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{CropImageTransform, FlipImageTransform, MultiImageTransform, RotateImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.examples.deeplogo.{MyImageRecordReader, MyMultiEpochIterator, MyRotateImageTransform, MyUIServer}
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels.VGG16
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions
import scala.collection.JavaConversions._

/**
  * Created by andlatel on 21/05/2017.
  */
class LogoClassification(val network: MultiLayerNetwork, conf: Configuration) {
  val log: Logger = LoggerFactory.getLogger(classOf[LogoClassification])

  def exec() = {
    log.info("Load data....")

    /**
      * Data Setup -> organize and limit data file paths:
      *  - mainPath = path to image files
      *  - fileSplit = define basic dataset split with limits on format
      *  - pathFilter = define additional file load filter to limit size and balance batch content
      * */
    val labelMaker = new ParentPathLabelGenerator

    //val mainPath = new File("d:\\Users\\andlatel\\Desktop\\images\\")
    //val mainPath = new File("d:\\Users\\andlatel\\Desktop\\Documents Project\\deeplogo\\dataset\\MyLogosExt_v2.0\\")

    //val mainPath = new File(System.getProperty("user.home"), "data/images/")

    //val mainPath = new File(System.getProperty("user.home"), "data/MyLogosExt_v2.0/")
    val mainPath = new File(System.getProperty("user.dir"), "../../../data/MyLogosCropped100/")
    //val mainPath = new File(System.getProperty("user.home"), "data/FlickrLogos-v2/classes/jpg/")

    val fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, conf.rng)
    //val fileSplitTest = new FileSplit(mainPathTest, NativeImageLoader.ALLOWED_FORMATS, conf.rng)

    val pathFilter = new RandomPathFilter(conf.rng, NativeImageLoader.ALLOWED_FORMATS, 0)
    //val pathFilter = new BalancedPathFilter(conf.rng, labelMaker, conf.numExamples, conf.numLabels, conf.maxPathPerLabels)


    /**
      * Data Setup -> train test split
      *  - inputSplit = define train and test split
      **/
    val inputSplit: Array[InputSplit] = fileSplit.sample(pathFilter, conf.numExamples * (conf.splitTrainTest), conf.numExamples * (1 - conf.splitTrainTest))
    //val inputSplitTest: Array[InputSplit] = fileSplitTest.sample(pathFilterTest, 2240, 0)
    val trainData = inputSplit(0)
    val testData  = inputSplit(1)

    /** Data Setup -> normalization
      *  - how to normalize images and generate large dataset to train on
      **/
    val scaler = new ImagePreProcessingScaler(0, 1)
    log.info("Build model....")
    // Uncomment below to try AlexNet. Note change height and width to at least 100
    // MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

    network.init()
    setListners()

    /**
      * Data Setup -> define how to load data into net:
      *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
      *  - dataIter = a generator that only loads one batch at a time into memory to save memory
      *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
      **/
    //val transform = new MultiImageTransform(conf.rng,
    //  new CropImageTransform(256),
    //  new FlipImageTransform())


    log.info("Train model....")
    val recordReader1 = new ImageRecordReader(conf.height, conf.width, conf.channels, labelMaker)
    val recordReader2 = new ImageRecordReader(conf.height, conf.width, conf.channels, labelMaker)

    recordReader1.initialize(trainData, null)
    val trainIter = new RecordReaderDataSetIterator(recordReader1, conf.batchSize, 1, conf.numLabels)
    trainIter.setPreProcessor(scaler)

    recordReader2.initialize(testData, null)
    val testIter = new RecordReaderDataSetIterator(recordReader2, conf.batchSize, 1, conf.numLabels)
    testIter.setPreProcessor(scaler)

    val trainMIter = new MultipleEpochsIterator(conf.epochs, trainIter, conf.nCores)
    //getImage[MultipleEpochsIterator](trainIter)
    network.fit(trainIter)

    var iter = 0
    var eval: Evaluation = null
    var force = false
    while (trainMIter.hasNext && !force) {
      network.fit(trainMIter.next)
      if (iter % 10 == 0) {
        System.out.println("Evaluate model at iter " + iter + " .... Good Test")
        eval = network.evaluate(testIter)
        System.out.println(eval.stats)
        testIter.reset()
        if(eval.accuracy() > 0.70)
          force=true

      }
      iter += 1
    }
    System.out.println("Model build complete")

    testIter.reset()
    var i = 0
    for(i <- 0 to 10) {
      val features = testIter.next().getFeatures()
      val start = System.currentTimeMillis()

      val out = network.output(features)

      val end = System.currentTimeMillis()

      System.out.println("Out Extraction:" + (end - start) + " ms")
    }

    if (conf.save) {
      log.info("Save model....")
      //val basePath = FilenameUtils.concat(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/")
      val basePath = "/home/deeplogo/"
      ModelSerializer.writeModel(network, basePath + "model.bin", true)
    }

    log.info("End.")

  }

  def setListners() = {
    val uiServer = MyUIServer.getInstance
    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    //Alternative: new FileStatsStorage(File), for saving and loading later
    val statsStorage = new InMemoryStatsStorage

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)

    //Then add the StatsListener to collect this information from the network, as it trains
    val listeners = Seq(new StatsListener(statsStorage), new ScoreIterationListener(conf.listenerFreq))
    network.setListeners(listeners)
  }

  def getImage[T <: DataSetIterator ](dataIter: T) = {
    while (dataIter.hasNext) {
      val batchDataSet = dataIter.next()
      batchDataSet.toList.map(ds=>{
        val res = toImage(ds.getFeatures)
        writeImage(res)
        try
          Thread.sleep(300) //1000 milliseconds is one second.
        catch {
          case ex: InterruptedException =>
            Thread.currentThread.interrupt()
        }
      })
    }
  }


  def toImage(matrix: INDArray): BufferedImage = {
    import java.awt.image.BufferedImage
    val img = new BufferedImage(conf.width, conf.height, BufferedImage.TYPE_INT_ARGB)
    val a = 255
    var p = 0

    for(row <- 0 to conf.height-1){
      for(col <- 0 to conf.width-1){
        val r = matrix.getColumn(0).getColumn(row).getInt(col)
        val g = matrix.getColumn(1).getColumn(row).getInt(col)
        val b = matrix.getColumn(2).getColumn(row).getInt(col)
        //val gray = matrix.getColumn(0).getColumn(row).getInt(col)
        //set the pixel value
        p = (a<<24) | (b<<16) | (g<<8) | r
        //p = (gray<<16) | (gray<<8) | gray
        img.setRGB(col, row, p)
      }
    }

    img
  }

  def writeImage(bi: BufferedImage) = {
    try {
      log.info("Write Image")
      val outputfile = new File("D:\\saved.jpg");
      ImageIO.write(bi, "png", outputfile);
    } catch {
      case e:Throwable => log.error("Error write image", e)
    }
  }

}
