package deeplogo

import net.CustomNet

/**
  * Created by andlatel on 20/05/2017.
  */
object Main extends App {

  val conf = new ConfigurationImpl()
  val net = new CustomNet(conf).createNet()

  new LogoClassification(net, conf).exec()

  //new Test(conf).test()
}
