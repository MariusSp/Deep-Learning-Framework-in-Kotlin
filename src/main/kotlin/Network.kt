import inputs.Input
import layers.Layer
import layers.ShapeChangingLayer
import structure.Tensor
import java.text.DecimalFormat
import java.util.*
import kotlin.collections.ArrayList
import kotlin.concurrent.schedule
import kotlin.math.roundToLong


class Network(vararg layers2: Layer) {
    private val layers = layers2.toMutableList()
    private val forwardTensors = mutableListOf<ArrayList<Tensor>>()
    private var backwardTensors = mutableListOf<ArrayList<Tensor>>()

    private var avgLoss = 0F
    private var entryCount = 0
    private var accuracy = 0F
    private var positive = 0
    private val dfLoss = DecimalFormat("#.###")
    private val dfAccuracy = DecimalFormat("#.##")
    private val df2 = DecimalFormat("#00")

    private fun forward(batchData: ArrayList<Tensor>) {
        forwardTensors[0] = batchData
        for ((i, layer) in layers.withIndex()) {
            layer.forward(forwardTensors[i], forwardTensors[i + 1])
        }
    }

    private fun backward(labels: ArrayList<Tensor>, lossFunctionDer: (ArrayList<Tensor>, ArrayList<Tensor>, ArrayList<Tensor>) -> Unit) {
        lossFunctionDer(forwardTensors.last(), labels, backwardTensors[0])
        for ((i, layer) in layers.asReversed().withIndex()) {
            layer.backward(backwardTensors[i], backwardTensors[i + 1], forwardTensors[forwardTensors.size - i - 2])
        }
    }

    private fun updateWeights(learningRate: Float, optimizer: (Float, ArrayList<Tensor>, Tensor) -> Unit) {
        layers.forEach {
            if (it is ShapeChangingLayer) {
                it.updateWeights(learningRate, optimizer)
            }
        }
    }

    private fun trainBatch(lossFunction: (ArrayList<Tensor>, ArrayList<Tensor>) -> Float, lossFunctionDer: (ArrayList<Tensor>, ArrayList<Tensor>, ArrayList<Tensor>) -> Unit, batchData: ArrayList<Tensor>, batchLabels: ArrayList<Tensor>) {
        forwardTensors[0] = batchData
        val tensorsBackwards = ArrayList(batchData.toMutableList())
        for (i in tensorsBackwards.indices) {
            tensorsBackwards[i] = tensorsBackwards[i].copy(transpose = true)
        }
        backwardTensors[backwardTensors.size - 1] = tensorsBackwards

        forward(batchData)
        val lossValue = lossFunction(forwardTensors.last(), batchLabels)
        backward(batchLabels, lossFunctionDer)

        //calculate loss
        entryCount += batchData.size
        avgLoss = avgLoss * (entryCount - 1) / entryCount + lossValue / entryCount

        //calculate accuracy
        forwardTensors.last().withIndex().forEach { (i, predictedLabel) ->
            if (predictedLabel.elements.indices.maxBy { value -> predictedLabel.elements[value] } == batchLabels[i].elements.indices.maxBy { value -> batchLabels[i].elements[value] }) {
                positive += 1
            }
        }
        accuracy = positive / entryCount.toFloat()
    }

    fun train(inputData: Input, epochs: Int, batchsize: Int, learningRate: Float, lossFunction: (ArrayList<Tensor>, ArrayList<Tensor>) -> Float, lossFunctionDer: (ArrayList<Tensor>, ArrayList<Tensor>, ArrayList<Tensor>) -> Unit, optimizer: (Float, ArrayList<Tensor>, Tensor) -> Unit) {
        println("dataset consists of " + inputData.input.size + " items with poolSize " + Arrays.toString(inputData.input[0].shape.axis))
        val batches = inputDataToBatches(inputData, batchsize)
        initializeLayers(batches.first().first)
        print("training with batchsize of ${batches.first().first.size}:")

        val startTime = System.currentTimeMillis()
        var epochCounter = 1
        val t = Timer("stats console output")
        t.schedule(1000, 1000) {
            if (entryCount != 0) {
                val runTime = (System.currentTimeMillis() - startTime)
                val epochsDone = ((epochCounter - 1) + entryCount / inputData.labels.size.toFloat())
                val runTimeLeft = (runTime / epochsDone * epochs).roundToLong() - runTime
                print("\r$entryCount/${inputData.labels.size}\tloss ${dfLoss.format(avgLoss)}\tacc ${dfAccuracy.format(accuracy * 100)}\t(${df2.format(runTimeLeft / 1000 / 60)}:${df2.format(runTimeLeft / 1000 % 60)})")
            }
        }
        while (epochCounter <= epochs) {
            val epochStartTime = System.currentTimeMillis()
            println("\nepoch $epochCounter/$epochs")
            for (batch in batches) {
                trainBatch(lossFunction, lossFunctionDer, batch.first, batch.second)
                updateWeights(learningRate, optimizer)
            }
            printEpochStats(epochStartTime)
            epochCounter++
        }
        t.cancel()
        printTrainingStats(startTime)
    }

    private fun printTrainingStats(startTime: Long) {
        val runTime = System.currentTimeMillis() - startTime
        print("\ntotal duration: ${df2.format(runTime / 1000 / 60)}:${df2.format(runTime / 1000 % 60)}")

    }

    private fun printEpochStats(epochStartTime: Long) {
        val runTime = System.currentTimeMillis() - epochStartTime
        print("\r$entryCount/$entryCount\tloss:${dfLoss.format(avgLoss)}\tacc:${dfAccuracy.format(accuracy * 100)}\t(${df2.format(runTime / 1000 / 60)}:${df2.format(runTime / 1000 % 60)})")
        entryCount = 0
        avgLoss = 0F
        accuracy = 0F
        positive = 0
    }

    private fun initializeLayers(inputBatch: ArrayList<Tensor>) {
        var shape = inputBatch.first().shape
        for (layer in layers) {
            val tensors = mutableListOf<Tensor>()
            if (layer is ShapeChangingLayer) {
                shape = layer.outputShape
                for (i in 0 until inputBatch.size) {
                    val tensor = Tensor(layer.outputShape, false)
                    tensors.add(tensor)
                }
            } else {
                for (i in 0 until inputBatch.size) {
                    val tensor = Tensor(shape, false)
                    tensors.add(tensor)
                }
            }
            forwardTensors.add(ArrayList(tensors))
            val tensorsBackwards = tensors.toMutableList()
            for (i in tensorsBackwards.indices) {
                tensorsBackwards[i] = tensorsBackwards[i].copy(transpose = true)
            }
            backwardTensors.add(0, ArrayList(tensorsBackwards))
        }
        forwardTensors.add(0, inputBatch)
        backwardTensors.add(inputBatch)
    }

    fun evaluate(evaluationData: Input) {
        val batches = inputDataToBatches(evaluationData, this.forwardTensors.first().size)
        println("\n\nevaluate on ${evaluationData.labels.size} items")
        for (batch in batches) {
            forward(batch.first)
            entryCount += batch.first.size
            forwardTensors.last().withIndex().forEach { (i, predictedLabel) ->
                if (predictedLabel.elements.indices.maxBy { value -> predictedLabel.elements[value] } == batch.second[i].elements.indices.maxBy { value -> batch.second[i].elements[value] }) {
                    positive += 1
                }
            }
            accuracy = positive / entryCount.toFloat()
            print("\r$entryCount/${evaluationData.labels.size} items - acc ${dfAccuracy.format(accuracy * 100)}")
        }
    }

    /**
     * processes Data to Batches (List of pairs(inputdata, label))
     */
    private fun inputDataToBatches(inputData: Input, batchsize: Int): List<Pair<ArrayList<Tensor>, ArrayList<Tensor>>> {
        val batches = emptyList<Pair<ArrayList<Tensor>, ArrayList<Tensor>>>().toMutableList()

        var i = 0
        while (i + batchsize <= inputData.input.size) {
            batches.add(Pair(ArrayList(inputData.input.subList(i, i + batchsize)), ArrayList(inputData.labels.subList(i, i + batchsize))))
            i += batchsize
        }
        return batches
    }
}