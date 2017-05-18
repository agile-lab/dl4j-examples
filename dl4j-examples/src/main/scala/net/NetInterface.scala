package net

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

/**
  * Created by andlatel on 17/05/2017.
  */
trait NetInterface {
    def createNet(): MultiLayerNetwork
}
