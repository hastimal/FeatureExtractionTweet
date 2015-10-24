package edu.umkc.group1

import java.io.{IOException, PrintStream}
import java.net.{InetAddress, Socket}

/**
 * Created by Mayanka on 08-Oct-15.
 * Modified by Hastimal on 10/23/2015.
 */

//134.193.136.213
object iOSConnector {

  def findIpAdd(): String = {
    val localhost = InetAddress.getLocalHost
    val localIpAddress = localhost.getHostAddress

    return localIpAddress
  }

  def getSocket(): Socket = {
    lazy val address: Array[Byte] = Array(10.toByte, 205.toByte, 1.toByte, 150.toByte)
    val ia = InetAddress.getByAddress(address)
    //lazy val sourceAddress: Array[Byte] = Array(10.toByte, 205.toByte, 1.toByte, 150.toByte)
    val oa = InetAddress.getLocalHost
    val socket = new Socket(ia, 1234, oa, 1234)
    socket
  }

  def sendCommandToRobot(string: String, socket: Socket) {
    // Simple server

    try {


      val out = new PrintStream(socket.getOutputStream)
      //val in = new DataInputStream(socket.getInputStream())

      out.print(string)
      out.flush()

      out.close()
      //in.close()

    }
    catch {
      case e: IOException =>
        e.printStackTrace()
    }
  }

}