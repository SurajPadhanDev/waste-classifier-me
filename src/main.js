import './style.css'
import { WasteClassifier } from './classifier.js'
import { CameraManager } from './camera.js'
import { UIManager } from './ui.js'

class WasteClassifierApp {
    constructor() {
        this.classifier = new WasteClassifier()
        this.camera = new CameraManager()
        this.ui = new UIManager()
        this.isProcessing = false
        this.predictionBuffer = []
        this.maxBufferSize = 10
        
        this.init()
    }

    async init() {
        try {
            // Initialize classifier
            await this.classifier.init()
            this.ui.updateModelStatus('Ready', 'ready')
            
            // Setup event listeners
            this.setupEventListeners()
            
            // Hide loading overlay
            this.ui.hideLoadingOverlay()
            
        } catch (error) {
            console.error('Failed to initialize app:', error)
            this.ui.updateModelStatus('Error', 'error')
            this.ui.hideLoadingOverlay()
        }
    }

    setupEventListeners() {
        // Camera controls
        document.getElementById('start-camera').addEventListener('click', () => this.startCamera())
        document.getElementById('stop-camera').addEventListener('click', () => this.stopCamera())
        
        // Image upload
        const uploadArea = document.getElementById('upload-area')
        const fileInput = document.getElementById('image-upload')
        
        uploadArea.addEventListener('click', () => fileInput.click())
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e))
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e))
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e))
    }

    async startCamera() {
        try {
            this.ui.updateCameraStatus('Starting...', 'loading')
            
            const stream = await this.camera.start()
            if (stream) {
                this.ui.updateCameraStatus('Active', 'active')
                this.ui.toggleCameraControls(true)
                this.ui.showCameraFeed()
                
                // Start processing frames
                this.processVideoFrames()
            }
        } catch (error) {
            console.error('Failed to start camera:', error)
            this.ui.updateCameraStatus('Error', 'error')
            alert('Failed to access camera. Please check permissions.')
        }
    }

    stopCamera() {
        this.camera.stop()
        this.ui.updateCameraStatus('Inactive', 'inactive')
        this.ui.toggleCameraControls(false)
        this.ui.hideCameraFeed()
        this.ui.hideLiveResult()
        this.predictionBuffer = []
    }

    async processVideoFrames() {
        if (!this.camera.isActive() || this.isProcessing) {
            if (this.camera.isActive()) {
                requestAnimationFrame(() => this.processVideoFrames())
            }
            return
        }

        this.isProcessing = true

        try {
            const imageData = this.camera.captureFrame()
            if (imageData) {
                const result = await this.classifier.predict(imageData)
                
                // Add to buffer for smoothing
                this.predictionBuffer.push(result)
                if (this.predictionBuffer.length > this.maxBufferSize) {
                    this.predictionBuffer.shift()
                }
                
                // Calculate averaged prediction
                const avgResult = this.calculateAverageResult(this.predictionBuffer)
                this.ui.updateLiveResult(avgResult)
            }
        } catch (error) {
            console.error('Error processing frame:', error)
        }

        this.isProcessing = false
        
        if (this.camera.isActive()) {
            requestAnimationFrame(() => this.processVideoFrames())
        }
    }

    calculateAverageResult(buffer) {
        if (buffer.length === 0) return null

        const classCount = {}
        let totalConfidence = 0

        buffer.forEach(result => {
            classCount[result.class] = (classCount[result.class] || 0) + 1
            totalConfidence += result.confidence
        })

        // Find most frequent class
        const mostFrequentClass = Object.keys(classCount).reduce((a, b) => 
            classCount[a] > classCount[b] ? a : b
        )

        return {
            class: mostFrequentClass,
            confidence: totalConfidence / buffer.length
        }
    }

    handleDragOver(e) {
        e.preventDefault()
        e.stopPropagation()
        e.currentTarget.classList.add('drag-over')
    }

    handleDrop(e) {
        e.preventDefault()
        e.stopPropagation()
        e.currentTarget.classList.remove('drag-over')
        
        const files = e.dataTransfer.files
        if (files.length > 0) {
            this.processUploadedFile(files[0])
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0]
        if (file) {
            this.processUploadedFile(file)
        }
    }

    async processUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.')
            return
        }

        try {
            this.ui.showUploadedImage(file)
            this.ui.showUploadResult({ class: 'Analyzing...', confidence: 0 })
            
            const imageData = await this.loadImageData(file)
            const result = await this.classifier.predict(imageData)
            
            this.ui.updateUploadResult(result)
            
        } catch (error) {
            console.error('Error processing uploaded file:', error)
            alert('Error processing image. Please try again.')
        }
    }

    loadImageData(file) {
        return new Promise((resolve, reject) => {
            const img = new Image()
            const canvas = document.createElement('canvas')
            const ctx = canvas.getContext('2d')
            
            img.onload = () => {
                canvas.width = 224
                canvas.height = 224
                ctx.drawImage(img, 0, 0, 224, 224)
                
                const imageData = ctx.getImageData(0, 0, 224, 224)
                resolve(imageData)
            }
            
            img.onerror = reject
            img.src = URL.createObjectURL(file)
        })
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WasteClassifierApp()
})