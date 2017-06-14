package org.deeplearning4j.examples.transferlearning.vgg16;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels.VGG16;

/**
 * @author susaneraly on 3/9/17.
 *
 * We use the transfer learning API to construct a new model based of vgg16
 * We will hold all layers but the very last one frozen and change the number of outputs in the last layer to
 * match our classification task.
 * In other words we go from where fc2 and predictions are vertex names in vgg16
 *  fc2 -> predictions (1000 classes)
 *  to
 *  fc2 -> predictions (5 classes)
 * The class "FitFromFeaturized" attempts to train this same architecture the difference being the outputs from the last
 * frozen layer is presaved and the fit is carried out on this featurized dataset.
 * When running multiple epochs this can save on computation time.
 */
@Slf4j
public class TransferGoogleNet {

    protected static final int numLabels = 8;
    protected static final long seed = 12345;

    //private static final int trainPerc = 80;
    private static final int batchSize = 10;
    private static final String featureExtractionLayer = "fc2";

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {



        File modelFile = new File("C:\\Users\\paolo\\Documents\\data\\googlenet_dl4j_inference.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        System.out.println(model.summary());


        //KerasModelImport.importKerasModelAndWeights(, "C:\\Users\\agilelab\\.dl4j\\trainedmodels\\resnet50\\resnet50_weights_th_dim_ordering_th_kernels.h5", false);


        //ZooModel zooModel = new VGG16();
        //Model net = zooModel.initPretrained(PretrainedType.IMAGENET);

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .learningRate(0.001)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        MultiLayerNetwork googleNetTransfer = new TransferLearning.Builder(model)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(model.getnLayers()-2) //the specified layer and below are "frozen"
            .removeLayersFromOutput(2)
            .addLayer(new DenseLayer.Builder().name("ffn2").nOut(1024).biasInit(0).dropOut(0.5).dist(new GaussianDistribution(0, 0.01)).build())
            .addLayer(
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(1024).nOut(numLabels)
                    .weightInit(WeightInit.DISTRIBUTION)
                    .dist(new NormalDistribution(0,0.2*(2.0/(4096+numLabels)))) //This weight init dist gave better results than Xavier
                    .activation(Activation.SOFTMAX).build())
            .build();
        System.out.println(googleNetTransfer.summary());


        //------------------------------------------------------

        int numExamples = 970;
        double splitTrainTest= 0.8;
        int height = 224;
        int width = 224;
        int channels = 3;
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        Random rng = new Random(43);
        int maxPathPerLabel = 100;
        int epochs = 2;
        int ncores = 6;


        File mainPath = new File("C:\\Users\\agilelab\\Documents\\MyLogosCropped100");
        File testPath = new File("C:\\Users\\agilelab\\Documents\\Flickr32_7classes");

        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit fileSplitTest = new FileSplit(testPath, NativeImageLoader.ALLOWED_FORMATS, rng);

        //BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathPerLabel);
        RandomPathFilter pathFilter = new RandomPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS,0 );



        InputSplit[] inputSplit = fileSplit.sample(pathFilter, numExamples * (splitTrainTest), numExamples * (1 - splitTrainTest));
        InputSplit trainData = inputSplit[0];
        InputSplit testData1 = inputSplit[1];
        InputSplit testData2 = fileSplitTest.sample(pathFilter,200)[0];

        ImageRecordReader recordReader1 = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader recordReader2 = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader recordReader3 = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator trainIter;
        DataSetIterator testIter;
        DataSetIterator testIter2;


        recordReader1.initialize(trainData, null);
        trainIter = new RecordReaderDataSetIterator(recordReader1, batchSize, 1, numLabels);
        trainIter.setPreProcessor(VGG16.getPreProcessor());

        recordReader2.initialize(testData1, null);
        testIter = new RecordReaderDataSetIterator(recordReader2, batchSize, 1, numLabels);
        testIter.setPreProcessor(VGG16.getPreProcessor());

        recordReader3.initialize(testData2, null);
        testIter2 = new RecordReaderDataSetIterator(recordReader3, batchSize, 1, numLabels);
        testIter2.setPreProcessor(VGG16.getPreProcessor());

        MultipleEpochsIterator trainMIter;
        trainMIter = new MultipleEpochsIterator(epochs, trainIter, ncores);

        //Dataset iterators
        //FlowerDataSetIterator.setup(batchSize,trainPerc);
        //DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
        //DataSetIterator testIter = FlowerDataSetIterator.testIterator();

        Evaluation eval;
        eval = googleNetTransfer.evaluate(testIter);
        System.out.println("Eval stats BEFORE fit on Good Test.....");
        System.out.println(eval.stats() + "\n");
        testIter.reset();

        eval = googleNetTransfer.evaluate(testIter2);
        System.out.println("Eval stats BEFORE fit on Bad Test.....");
        System.out.println(eval.stats() + "\n");
        testIter2.reset();

        int iter = 0;
        while(trainMIter.hasNext()) {
            googleNetTransfer.fit(trainMIter.next());
            if (iter % 10 == 0) {
                System.out.println("Evaluate model at iter "+iter +" .... Good Test");
                eval = googleNetTransfer.evaluate(testIter);
                System.out.println(eval.stats());
                testIter.reset();

                System.out.println("Evaluate model at iter "+iter +" .... Bad Test");
                eval = googleNetTransfer.evaluate(testIter2);
                System.out.println(eval.stats());
                testIter2.reset();
            }
            iter++;
        }

        System.out.println("Model build complete");

        File locationToSave = new File("MyLogoGraphGoogleNet.zip");
        boolean saveUpdater = false;
        ModelSerializer.writeModel(googleNetTransfer, locationToSave, saveUpdater);

        System.out.println("Model saved");


        for(int i=0; i < 10; i++) {
            INDArray features = testIter2.next().getFeatures();
            long start = System.currentTimeMillis();

            INDArray out = googleNetTransfer.output(features);

            long end = System.currentTimeMillis();

            System.out.println("Out Extraction:" + (end - start) + " ms");
        }
    }
}
