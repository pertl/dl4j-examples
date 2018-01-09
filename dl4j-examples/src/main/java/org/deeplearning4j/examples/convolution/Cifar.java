package org.deeplearning4j.examples.convolution;

import javax.swing.*;
import java.io.File;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Cifar {

    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("user.dir"), "/");
    private static final Logger log = LoggerFactory.getLogger(Cifar.class);

    private static final int height = 32;
    private static final int width = 32;
    private static final int channels = 3;
    private static final int numLabels = CifarLoader.NUM_LABELS;
    private static final int numSamples = 50000;
    private static final int batchSize = 100;
    private static final int iterations = 1;
    private static final int freIterations = 50;
    private static final int seed = 123;
    private static final boolean preProcessCifar = false;//use Zagoruyko's preprocess for Cifar
    private static final int epochs = 50;

    private Cifar() { }

    public static void main(String[] args) {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
        //train model and eval model
        MultiLayerNetwork model = Cifar.trainModelByCifarWithNet();//ignore
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(freIterations));

        CifarDataSetIterator cifar = new CifarDataSetIterator(batchSize, numSamples,
            new int[]{height, width, channels}, preProcessCifar, true);
        CifarDataSetIterator cifarEval = new CifarDataSetIterator(batchSize, 10000,
            new int[]{height, width, channels}, preProcessCifar, false);

        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch=====================" + i);
            model.fit(cifar);
        }

        log.info("=====eval model========");
        Evaluation eval = new Evaluation(cifarEval.getLabels());
        while (cifarEval.hasNext()) {
            DataSet testDS = cifarEval.next(batchSize);
            INDArray output = model.output(testDS.getFeatureMatrix());
            eval.eval(testDS.getLabels(), output);
        }
        System.out.println(eval.stats());

        Cifar.testModelByUnkownImage(model);
        Cifar.saveModel(model, "trainModelByCifarWithAlexNet_model.json");
    }

    private static MultiLayerNetwork trainModelByCifarWithNet() {
        log.info("this is Net for the cifar");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .cacheMode(CacheMode.DEVICE)
            .updater(Updater.ADAM)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .l1(1e-4)
            .regularization(true)
            .l2(5 * 1e-4)
            .list()
            .layer(0, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn1").convolutionMode(ConvolutionMode.Same)
                .nIn(3).nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)//.learningRateDecayPolicy(LearningRatePolicy.Step)
                .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
            .layer(1, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn2").convolutionMode(ConvolutionMode.Same)
                .nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
            .layer(2, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2, 2}).name("maxpool2").build())

            .layer(3, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn3").convolutionMode(ConvolutionMode.Same)
                .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
            .layer(4, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn4").convolutionMode(ConvolutionMode.Same)
                .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())

            .layer(5, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("cnn5").convolutionMode(ConvolutionMode.Same)
                .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
            .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("cnn6").convolutionMode(ConvolutionMode.Same)
                .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())

            .layer(7, new ConvolutionLayer.Builder(new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0}).name("cnn7").convolutionMode(ConvolutionMode.Same)
                .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
            .layer(8, new ConvolutionLayer.Builder(new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0}).name("cnn8").convolutionMode(ConvolutionMode.Same)
                .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
            .layer(9, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2, 2}).name("maxpool8").build())

            .layer(10, new DenseLayer.Builder().name("ffn1").nOut(1024).learningRate(1e-3).biasInit(1e-3).biasLearningRate(1e-3 * 2).build())
            .layer(11, new DropoutLayer.Builder().name("dropout1").dropOut(0.2).build())
            .layer(12, new DenseLayer.Builder().name("ffn2").nOut(1024).learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
            .layer(13, new DropoutLayer.Builder().name("dropout2").dropOut(0.2).build())
            .layer(14, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    private static void saveModel(MultiLayerNetwork model, String fileName) {
        File locationModelFile = new File(DATA_PATH + fileName);
        try {
            ModelSerializer.writeModel(model, locationModelFile, false);
        } catch (Exception e) {
            log.error("Saving model is not success !", e);
        }
    }

    private static void testModelByUnkownImage(MultiLayerNetwork model) {
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        String filename = "";
        if (ret == JFileChooser.APPROVE_OPTION) {
            File file = fc.getSelectedFile();
            filename = file.getAbsolutePath();
        }
        AnalysisUnkownImage ui = new AnalysisUnkownImage(filename, model, width, height);
        ui.showGUI();
    }

}

