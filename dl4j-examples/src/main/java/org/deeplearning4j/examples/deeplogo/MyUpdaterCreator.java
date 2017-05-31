package org.deeplearning4j.examples.deeplogo;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.LayerUpdater;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;

/**
 * Created by andlatel on 25/05/2017.
 */
public class MyUpdaterCreator {

    private MyUpdaterCreator() {}

    public static org.deeplearning4j.nn.api.Updater getUpdater(Model layer) {
        /*if (layer instanceof MyMultiLayerNetwork) {
            return new MyMultiLayerUpdater((MyMultiLayerNetwork) layer);
        } else {*/
            return new LayerUpdater();
    /*    }*/
    }
}
