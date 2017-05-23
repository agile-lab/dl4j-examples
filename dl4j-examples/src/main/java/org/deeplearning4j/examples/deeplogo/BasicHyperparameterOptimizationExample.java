//package org.deeplearning4j.examples.deeplogo;
//
///**
// * Created by andlatel on 22/05/2017.
// */
//
//import deeplogo.Configuration;
//import deeplogo.ConfigurationImpl;
//import org.datavec.api.io.filters.BalancedPathFilter;
//import org.datavec.api.io.labels.ParentPathLabelGenerator;
//import org.datavec.api.split.FileSplit;
//import org.datavec.api.split.InputSplit;
//import org.datavec.image.loader.NativeImageLoader;
//import org.datavec.image.recordreader.ImageRecordReader;
//import org.deeplearning4j.arbiter.DL4JConfiguration;
//import org.deeplearning4j.arbiter.MultiLayerSpace;
//import org.deeplearning4j.arbiter.data.DataSetIteratorProvider;
//import org.deeplearning4j.arbiter.layers.ConvolutionLayerSpace;
//import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
//import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
//import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
//import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
//import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
//import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
//import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
//import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
//import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
//import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
//import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
//import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
//import org.deeplearning4j.arbiter.optimize.candidategenerator.RandomSearchGenerator;
//import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
//import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
//import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
//import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
//import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
//import org.deeplearning4j.arbiter.optimize.ui.ArbiterUIServer;
//import org.deeplearning4j.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener;
//import org.deeplearning4j.arbiter.saver.local.multilayer.LocalMultiLayerNetworkSaver;
//import org.deeplearning4j.arbiter.scoring.multilayer.TestSetAccuracyScoreFunction;
//import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
//import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
//import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
//import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
//import org.deeplearning4j.nn.api.OptimizationAlgorithm;
//import org.deeplearning4j.nn.conf.inputs.InputType;
//import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
//import scala.Array;
//
//import java.io.File;
//import java.util.List;
//import java.util.concurrent.TimeUnit;
//
//
///**
// * This is a basic hyperparameter optimization example using Arbiter to conduct random search on two network hyperparameters.
// * The two hyperparameters are learning rate and layer size, and the search is conducted for a simple multi-layer perceptron
// * on MNIST data.
// *
// * Note that this example has a UI, but it (currently) does not start automatically.
// * By default, the UI is accessible at http://localhost:8080/arbiter
// *
// * @author Alex Black
// */
//public class BasicHyperparameterOptimizationExample {
//
//
//    public static void main(String[] args) throws Exception {
//        Configuration conf = new ConfigurationImpl();
//
//
//        //First: Set up the hyperparameter configuration space. This is like a MultiLayerConfiguration, but can have either
//        // fixed values or values to optimize, for each hyperparameter
//
//        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
//        ParameterSpace<Integer> layerSizeHyperparam = new IntegerParameterSpace(16,256);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)
//
//        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
//            //These next few options: fixed values for all models
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .iterations(1)
//            .regularization(true)
//            .l2(0.0001)
//            .learningRate(learningRateHyperparam)
//            .addLayer( new ConvolutionLayerSpace.Builder().nIn(conf.channels()).nOut(layerSizeHyperparam).kernelSize(3,3).stride(1,1).padding(1,1).activation("relu").build())
//            .addLayer( new DenseLayerSpace.Builder().activation("relu").nOut(layerSizeHyperparam).build())
//            .addLayer( new OutputLayerSpace.Builder().nOut(conf.numLabels()).activation("softmax").lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build())
//            .pretrain(false)
//            .backprop(true)
//            .setInputType(InputType.convolutional(conf.height(), conf.width(), conf.channels()))
//            .build();
//
//
//        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
//        //File mainPath = new File(System.getProperty("user.home"), "d:\\Users\\andlatel\\Desktop\\MyLogos\\");
//        File mainPath = new File("d:\\Users\\andlatel\\Desktop\\MyLogos\\");
//        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, conf.rng());
//        BalancedPathFilter pathFilter = new BalancedPathFilter(conf.rng(), labelMaker, 2240, conf.numLabels(), 70);
//
//        InputSplit[] inputSplit = fileSplit.sample(pathFilter, conf.numExamples() * (conf.splitTrainTest()), (1 - conf.splitTrainTest()));
//        //val inputSplitTest: Array[InputSplit] = fileSplitTest.sample(pathFilterTest, 2240, 0);
//        InputSplit trainData = inputSplit[0];
//        InputSplit testData  = inputSplit[1];
//
//        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
//
//        ImageRecordReader recordReader = new ImageRecordReader(conf.height(), conf.width(), conf.channels(), labelMaker);
//        ImageRecordReader recordReaderTest = new ImageRecordReader(conf.height(), conf.width(), conf.channels(), labelMaker);
//        recordReader.initialize(trainData, null);
//        recordReaderTest.initialize(testData, null);
//
//        DataSetIterator dataIter   = new RecordReaderDataSetIterator(recordReader, conf.batchSize(), 1, conf.numLabels());
//        DataSetIterator dataIterTest   = new RecordReaderDataSetIterator(recordReaderTest, conf.batchSize(), 1, conf.numLabels());
//        dataIter.setPreProcessor(scaler);
//        dataIterTest.setPreProcessor(scaler);
//
//        //Now: We need to define a few configuration options
//        // (a) How are we going to generate candidates? (random search or grid search)
//        CandidateGenerator<DL4JConfiguration> candidateGenerator = new RandomSearchGenerator<>(hyperparameterSpace);    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);
//
//        // (b) How are going to provide data? For now, we'll use a simple built-in data provider for DataSetIterators
//        int nTrainEpochs = 10;
//
//        //DataSetIterator mnistTrain = new MultipleEpochsIterator(nTrainEpochs, new MnistDataSetIterator(64,true,12345));
//        DataSetIterator mnistTrain = new MultipleEpochsIterator(nTrainEpochs, dataIter);
//        DataSetIterator mnistTest = new MnistDataSetIterator(64,false,12345);
//        DataProvider<DataSetIterator> dataProvider = new DataSetIteratorProvider(mnistTrain, mnistTest);
//
//        // (c) How we are going to save the models that are generated and tested?
//        //     In this example, let's save them to disk the working directory
//        //     This will result in examples being saved to arbiterExample/0/, arbiterExample/1/, arbiterExample/2/, ...
//        String baseSaveDirectory = "arbiterExample/";
//        File f = new File(baseSaveDirectory);
//        if(f.exists()) f.delete();
//        f.mkdir();
//
//        ResultSaver<DL4JConfiguration,MultiLayerNetwork,Object> modelSaver = new LocalMultiLayerNetworkSaver<>(baseSaveDirectory);
//
//        // (d) What are we actually trying to optimize?
//        //     In this example, let's use classification accuracy on the test set
//        ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction = new TestSetAccuracyScoreFunction();
//
//        // (e) When should we stop searching? Specify this with termination conditions
//        //     For this example, we are stopping the search at 15 minutes or 20 candidates - whichever comes first
//        TerminationCondition[] terminationConditions = {new MaxTimeCondition(15, TimeUnit.MINUTES), new MaxCandidatesCondition(20)};
//
//
//        //Given these configuration options, let's put them all together:
//        OptimizationConfiguration<DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object> configuration
//            = new OptimizationConfiguration.Builder<DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object>()
//            .candidateGenerator(candidateGenerator)
//            .dataProvider(dataProvider)
//            .modelSaver(modelSaver)
//            .scoreFunction(scoreFunction)
//            .terminationConditions(terminationConditions)
//            .build();
//
//        //And set up execution locally on this machine:
//        IOptimizationRunner<DL4JConfiguration,MultiLayerNetwork,Object> runner;
//        runner = new LocalOptimizationRunner<>(configuration, new MultiLayerNetworkTaskCreator<>());
//
//
//        //Start the UI
//        //ArbiterUIServer server = ArbiterUIServer.getInstance();
//        //runner.addListeners(new UIOptimizationRunnerStatusListener(server));
//
//
//        //Start the hyperparameter optimization
//        runner.execute();
//
//
//        //Print out some basic stats regarding the optimization procedure
//        StringBuilder sb = new StringBuilder();
//        sb.append("Best score: ").append(runner.bestScore()).append("\n")
//            .append("Index of model with best score: ").append(runner.bestScoreCandidateIndex()).append("\n")
//            .append("Number of configurations evaluated: ").append(runner.numCandidatesCompleted()).append("\n");
//        System.out.println(sb.toString());
//
//
//        //Get all results, and print out details of the best result:
//        int indexOfBestResult = runner.bestScoreCandidateIndex();
//        List<ResultReference<DL4JConfiguration,MultiLayerNetwork,Object>> allResults = runner.getResults();
//
//        OptimizationResult<DL4JConfiguration,MultiLayerNetwork,Object> bestResult = allResults.get(indexOfBestResult).getResult();
//        MultiLayerNetwork bestModel = bestResult.getResult();
//
//        System.out.println("\n\nConfiguration of best model:\n");
//        System.out.println(bestModel.getLayerWiseConfigurations().toJson());
//
//
//        //Note: UI server will shut down once execution is complete, as JVM will exit
//        //So do a Thread.sleep(1 minute) to keep JVM alive, so that network configurations can be viewed
//        Thread.sleep(60000);
//        System.exit(0);
//    }
//
//}
