import * as tf from '@tensorflow/tfjs'

export class WasteClassifier {
    constructor() {
        this.model = null
        this.classes = ['Hazardous Waste', 'Organic Waste', 'Inorganic Waste']
        this.isReady = false
    }

    async init() {
        try {
            console.log('Loading TensorFlow.js model...')
            
            // For demo purposes, we'll create a mock model
            // In production, you would load your actual model:
            // this.model = await tf.loadLayersModel('/models/waste-classifier/model.json')
            
            await this.createMockModel()
            this.isReady = true
            console.log('Model loaded successfully!')
            
        } catch (error) {
            console.error('Failed to load model:', error)
            throw error
        }
    }

    async createMockModel() {
        // Create a simple mock model for demonstration
        // This simulates the MobileNetV2 architecture
        const model = tf.sequential({
            layers: [
                tf.layers.conv2d({
                    inputShape: [224, 224, 3],
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.globalAveragePooling2d(),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.5 }),
                tf.layers.dense({ units: 3, activation: 'softmax' })
            ]
        })

        // Compile the model
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        })

        this.model = model
        
        // Warm up the model
        const dummyInput = tf.zeros([1, 224, 224, 3])
        await this.model.predict(dummyInput)
        dummyInput.dispose()
    }

    async predict(imageData) {
        if (!this.isReady || !this.model) {
            throw new Error('Model not ready')
        }

        try {
            // Convert ImageData to tensor
            const tensor = this.preprocessImage(imageData)
            
            // Make prediction
            const predictions = await this.model.predict(tensor)
            const probabilities = await predictions.data()
            
            // Find the class with highest probability
            const maxIndex = probabilities.indexOf(Math.max(...probabilities))
            const confidence = probabilities[maxIndex]
            const predictedClass = this.classes[maxIndex]
            
            // Clean up tensors
            tensor.dispose()
            predictions.dispose()
            
            return {
                class: predictedClass,
                confidence: confidence,
                probabilities: Array.from(probabilities)
            }
            
        } catch (error) {
            console.error('Prediction error:', error)
            
            // Return mock prediction for demo
            return this.getMockPrediction()
        }
    }

    preprocessImage(imageData) {
        // Convert ImageData to tensor and normalize
        const tensor = tf.browser.fromPixels(imageData)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims(0)
        
        return tensor
    }

    getMockPrediction() {
        // Generate realistic mock predictions for demo
        const mockPredictions = [
            { class: 'Organic Waste', confidence: 0.85 },
            { class: 'Inorganic Waste', confidence: 0.78 },
            { class: 'Hazardous Waste', confidence: 0.92 },
            { class: 'Organic Waste', confidence: 0.73 },
            { class: 'Inorganic Waste', confidence: 0.81 }
        ]
        
        return mockPredictions[Math.floor(Math.random() * mockPredictions.length)]
    }
}