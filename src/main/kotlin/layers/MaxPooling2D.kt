package layers

import structure.Shape
import structure.Tensor

class MaxPooling2D(val inputShape: Shape, private val poolSize: Shape, batchsize: Int) : ShapeChangingLayer {
    override val outputShape = Shape(inputShape.axis[0] / poolSize.axis[0] * inputShape.axis[1] / poolSize.axis[1], 1)
    private val masks = arrayListOf<IntArray>()
    private val siteY = poolSize.axis[0]
    private val siteX = poolSize.axis[1]

    private var indices = IntRange(0, batchsize - 1).toSet()

    init {
        if (inputShape.axis[0] % poolSize.axis[0] != 0 || inputShape.axis[1] % poolSize.axis[1] != 0) {
            throw IllegalArgumentException("pooling filter size doesnt fit on input layer!")
        }
        for (i in 0 until batchsize) {
            masks.add(IntArray(inputShape.volume) { -1 })
        }
    }

    override fun updateWeights(learningRate: Float, optimizer: (Float, ArrayList<Tensor>, Tensor) -> Unit) {
    }

    override fun forward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>) {
        indices.parallelStream().parallel().forEach { index ->

            val poolingArray = FloatArray(inputShape.axis[0] / poolSize.axis[0] * inputShape.axis[1] / poolSize.axis[1]) { 0F }
            val poolingArrayIndex = IntArray(inputShape.axis[0] / poolSize.axis[0] * inputShape.axis[1] / poolSize.axis[1]) { 0 }

            // alles ist noch row major
            for ((i, value) in inTensors[index].elements.withIndex()) {
                //val row = inTensors[index].rowOf(i)  shape von inputtensor ist immer (x,1) daher müssen wir ja uach den tatsächlien inputshapoe übergeben, daher geht dein rowof .. nicht
                val row = i / (inputShape.axis[0])
                //val column = inTensors[index].columnOf(i) same
                val column = i % inputShape.axis[0]
                //print("$i $row $column ")
                val poolIndex = column / siteY + (row / siteX) * inputShape.axis[0] / poolSize.axis[0] //hier muss du ja die Anzahl der Spalten der outputmatrix nehmen
                //println(poolIndex)
                if (value > poolingArray[poolIndex]) {
                    poolingArray[poolIndex] = value
                    masks[index][poolingArrayIndex[poolIndex]] = -1
                    poolingArrayIndex[poolIndex] = i
                    masks[index][i] = poolIndex
                }
            }
            outTensors[index].elements = poolingArray
        }
    }

    override fun backward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>, forwardInTensors: ArrayList<Tensor>) {
        indices.parallelStream().parallel().forEach { index ->
            for ((i, value) in masks[index].withIndex()) {
                val row = i / (inputShape.axis[0])
                val column = i % inputShape.axis[0]
                val poolIndex = column / siteY + (row / siteX) * inputShape.axis[0] / poolSize.axis[0]
                if (value != -1) {
                    outTensors[index].elements[i] = inTensors[index].elements[poolIndex] // der index i wirft bei mir arrayoutofbounds weil hier ja der outtensor (und maske) ja 4 mal so groß ist wie der inputtensor
                } else {
                    outTensors[index].elements[i] = 0F
                }
            }
        }
    }
}