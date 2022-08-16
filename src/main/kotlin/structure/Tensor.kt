package structure

class Tensor(val shape: Shape, initializeElementsRandom: Boolean, initializeValue: Float = 0F) {
    var elements = FloatArray(shape.volume) { initializeValue }

    init {
        if (initializeElementsRandom) {
            val initRange = (-1000..1000)
            for (i in elements.indices) {
                elements[i] = (initRange.random() / 1000F)
            }
        }
    }

    fun slowMatrixMul(inTensorB: Tensor, outTensor: Tensor) {
        if (this.shape.axis[1] != inTensorB.shape.axis[0] || this.shape.axis[0] != outTensor.shape.axis[0] || inTensorB.shape.axis[1] != outTensor.shape.axis[1]) {
            throw Exception("Mismatchin poolSize on TensorMullt")
        }
        for (i in 0 until outTensor.shape.axis[1]) {
            for (j in 0 until outTensor.shape.axis[0]) {
                outTensor.elements[j + i * outTensor.shape.axis[0]] = 0F
                for (k in 0 until this.shape.axis[1]) {
                    outTensor.elements[j + i * outTensor.shape.axis[0]] += this.elements[j + k * this.shape.axis[0]] * inTensorB.elements[k + i * inTensorB.shape.axis[0]]
                }
            }
        }
    }

    fun transpose(outTensor: Tensor) {
        if (outTensor.shape.axis[1] > 1 && outTensor.shape.axis[0] > 1) {
            for (i in 0 until outTensor.shape.axis[1]) {
                for (j in 0 until outTensor.shape.axis[0]) {
                    outTensor.elements[j + i * outTensor.shape.axis[0]] = this.elements[i + j * this.shape.axis[0]]
                }
            }
        } else {
            outTensor.elements = this.elements.clone()
        }
    }

    fun copy(transpose: Boolean): Tensor {
        val tensor: Tensor = if (transpose) {
            Tensor(shape.transpose(), false)
        } else {
            Tensor(shape, false)
        }
        for ((i, element) in elements.withIndex()) {
            tensor.elements[i] = element
        }
        return tensor
    }

    fun columnOf(index: Int): Int {
        var iColumn = (index % shape.axis[1])
        if (iColumn == -1) {
            iColumn = shape.axis[1] - 1
        }
        return iColumn
    }

    fun rowOf(index: Int): Int {
        return index / shape.axis[1]

    }

    fun get1dIndex(x: Int, y: Int): Int {
        return y * shape.axis[1] + x
    }
}