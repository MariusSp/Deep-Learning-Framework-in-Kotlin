package layers

import structure.Shape
import structure.Tensor

class Conv2D(private val filtershape: Shape, private val shape3d: Shape, batchsize: Int) : ShapeChangingLayer {
    override val outputShape = Shape(filtershape.axis[3] * (shape3d.axis[0] - filtershape.axis[0] + 1) * (shape3d.axis[1] - filtershape.axis[1] + 1), 1)
    var filter = Tensor(filtershape, true)
    private var derFilter = ArrayList(emptyList<Tensor>().toMutableList())
    private var bias = Tensor(Shape(filtershape.axis[3]), true)
    private var indices = IntRange(0, batchsize - 1).toSet()
    private var biasupdates = ArrayList(emptyList<Tensor>().toMutableList())

    init {
        val filterList = ArrayList(emptyList<Tensor>().toMutableList())
        val biasList = ArrayList(emptyList<Tensor>().toMutableList())

        for (i in 1..batchsize) {
            filterList.add(Tensor(filtershape, initializeElementsRandom = false))
            biasList.add(Tensor(Shape(filtershape.axis[3]), false))
        }

        derFilter = filterList
        biasupdates = biasList
    }

    override fun forward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>) {
        if (outTensors.size != inTensors.size)
            throw Exception("Tensorlistshape mismatch")
        indices.parallelStream().parallel().forEach { forwardconv(inTensors[it], outTensors[it], filter) }
    }

    override fun backward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>, forwardInTensors: ArrayList<Tensor>) {
        indices.parallelStream().parallel().forEach { backwardpass(inTensors[it], outTensors[it]) }
        indices.parallelStream().parallel().forEach { computedeltas(inTensors[it], derFilter[it], forwardInTensors[it], biasupdates[it]) }
    }

    override fun updateWeights(learningRate: Float, optimizer: (Float, ArrayList<Tensor>, Tensor) -> Unit) {
        optimizer(learningRate, derFilter, filter)
        optimizer(learningRate, biasupdates, bias)
    }

    fun backwardpass(inTensor: Tensor, outTensor: Tensor) {
        var outindex: Int
        var filterindex: Int
        var inindex: Int
        for (channel in 0 until shape3d.axis[2]) {
            for (xpos in 0 until shape3d.axis[1]) {
                for (ypos in 0 until shape3d.axis[0]) {
                    outindex = channel * shape3d.axis[1] * shape3d.axis[0] + xpos * shape3d.axis[0] + ypos
                    outTensor.elements[outindex] = 0F
                    for (currentfilter in 0 until filtershape.axis[3]) {
                        for (filposx in 0 until filtershape.axis[1]) {
                            for (filposy in 0 until filtershape.axis[0]) {
                                filterindex = currentfilter * filtershape.axis[2] * filtershape.axis[1] * filtershape.axis[0] + channel * filtershape.axis[1] * filtershape.axis[0] + filposx * filtershape.axis[0] + filposy
                                inindex = currentfilter * (shape3d.axis[1] - filtershape.axis[1] + 1) * (shape3d.axis[0] - filtershape.axis[0] + 1) + (xpos - filposx) * (shape3d.axis[0] - filtershape.axis[0] + 1) + ypos - filposy
                                if ((xpos - filposx) >= 0 && (xpos - filposx) <= (shape3d.axis[1] - filtershape.axis[1]) && (ypos - filposy) >= 0 && (ypos - filposy) <= (shape3d.axis[0] - filtershape.axis[0])) {
                                    outTensor.elements[outindex] += inTensor.elements[inindex] * filter.elements[filterindex]
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fun computedeltas(inTensor: Tensor, outTensor: Tensor, forwardinTensor: Tensor, biasupdate: Tensor) {
        var filterindex: Int
        var inputindex: Int
        var dyindex: Int
        for (currentfilter in 0 until filtershape.axis[3]) {
            for (channel in 0 until filtershape.axis[2]) {
                for (xpos in 0 until filtershape.axis[1]) {
                    for (ypos in 0 until filtershape.axis[0]) {
                        filterindex = currentfilter * filtershape.axis[2] * filtershape.axis[1] * filtershape.axis[0] + channel * filtershape.axis[1] * filtershape.axis[0] + xpos * filtershape.axis[0] + ypos

                        outTensor.elements[filterindex] = 0F
                        for (sumposx in 0..(shape3d.axis[1] - filtershape.axis[1])) {
                            for (sumposy in 0..(shape3d.axis[0] - filtershape.axis[0])) {
                                inputindex = channel * shape3d.axis[1] * shape3d.axis[0] + (xpos + sumposx) * shape3d.axis[0] + ypos + sumposy
                                dyindex = currentfilter * (shape3d.axis[1] - filtershape.axis[1] + 1) * (shape3d.axis[0] - filtershape.axis[0] + 1) + sumposx * (shape3d.axis[0] - filtershape.axis[0] + 1) + sumposy
                                //println("$filterindex $inputindex $dyindex")
                                outTensor.elements[filterindex] += forwardinTensor.elements[inputindex] * inTensor.elements[dyindex]

                            }
                        }
                    }
                }
            }
        }

        for (currentfilter in 0 until filtershape.axis[3]) {
            biasupdate.elements[currentfilter] = 0F
            for (xpos in 0..(shape3d.axis[1] - filtershape.axis[1])) {
                for (ypos in 0..(shape3d.axis[0] - filtershape.axis[0])) {
                    inputindex = currentfilter * (shape3d.axis[1] - filtershape.axis[1]) * (shape3d.axis[0] - filtershape.axis[0]) + xpos * (shape3d.axis[0] - filtershape.axis[0]) + ypos
                    biasupdate.elements[currentfilter] += inTensor.elements[inputindex]
                }
            }
        }
    }

    private fun forwardconv(inTensor: Tensor, outTensor: Tensor, filter: Tensor) {
        val vol2d = (shape3d.axis[1] - filtershape.axis[1] + 1) * (shape3d.axis[0] - filtershape.axis[0] + 1)
        var outindex: Int
        var inindex: Int
        var filterindex: Int
        //var save1 = 0F
        //var save2 =0F
        for (currentfilter in 0 until filtershape.axis[3]) {
            // iteration über output spalten
            for (xpos in 0..(shape3d.axis[1] - filtershape.axis[1])) {
                //iteration über output zeilen
                for (ypos in 0..(shape3d.axis[0] - filtershape.axis[0])) {
                    // bis hier interation über den output tensor, dann faltung:
                    outindex = currentfilter * vol2d + xpos * (shape3d.axis[0] - filtershape.axis[0] + 1) + ypos
                    outTensor.elements[outindex] = bias.elements[currentfilter]
                    // interatin über filterspalten
                    for (filposx in 0 until filtershape.axis[1]) {
                        // interation über filterzeilen
                        for (filposy in 0 until filtershape.axis[0]) {
                            // interatoin über inputchannels
                            for (c in 0 until filtershape.axis[2]) {
                                inindex = c * shape3d.axis[0] * shape3d.axis[1] + (xpos + filposx) * shape3d.axis[0] + (ypos + filposy)
                                filterindex = currentfilter * (filtershape.axis[0] * filtershape.axis[1] * filtershape.axis[2]) + c * (filtershape.axis[0] * filtershape.axis[1]) + filposx * filtershape.axis[0] + filposy
                                //save1 = inTensor.elements[inindex]
                                //save2 = filter[currentfilter].elements[filterindex]
                                //print("$outindex $inindex $filterindex $save1 $save2")
                                outTensor.elements[outindex] += inTensor.elements[inindex] * filter.elements[filterindex]
                            }
                        }
                    }
                }
            }

        }
    }
}