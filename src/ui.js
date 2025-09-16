export class UIManager {
    constructor() {
        this.disposalInfo = {
            'Organic Waste': {
                emoji: '🌱',
                message: 'Dispose in organic waste bin. This waste is biodegradable and can be composted.',
                color: '#22c55e'
            },
            'Hazardous Waste': {
                emoji: '☠️',
                message: 'Handle with care! Dispose at designated hazardous waste facility.',
                color: '#ef4444'
            },
            'Inorganic Waste': {
                emoji: '♻️',
                message: 'Great! This item can be recycled. Clean before disposing in recycling bin.',
                color: '#3b82f6'
            }
        }
    }

    updateModelStatus(status, type) {
        const statusElement = document.getElementById('model-status')
        statusElement.textContent = status
        statusElement.className = `status-value ${type}`
    }

    updateCameraStatus(status, type) {
        const statusElement = document.getElementById('camera-status')
        statusElement.textContent = status
        statusElement.className = `status-value ${type}`
    }

    toggleCameraControls(isActive) {
        const startBtn = document.getElementById('start-camera')
        const stopBtn = document.getElementById('stop-camera')
        
        startBtn.disabled = isActive
        stopBtn.disabled = !isActive
    }

    showCameraFeed() {
        const video = document.getElementById('camera-feed')
        const placeholder = document.getElementById('camera-placeholder')
        
        video.style.display = 'block'
        placeholder.style.display = 'none'
    }

    hideCameraFeed() {
        const video = document.getElementById('camera-feed')
        const placeholder = document.getElementById('camera-placeholder')
        
        video.style.display = 'none'
        placeholder.style.display = 'flex'
    }

    updateLiveResult(result) {
        if (!result) return

        const resultCard = document.getElementById('live-result')
        const classElement = document.getElementById('live-class')
        const confidenceElement = document.getElementById('live-confidence')
        const confidenceFill = document.getElementById('live-confidence-fill')

        resultCard.style.display = 'block'
        classElement.textContent = result.class
        confidenceElement.textContent = `${(result.confidence * 100).toFixed(1)}%`
        
        // Update confidence bar
        confidenceFill.style.width = `${result.confidence * 100}%`
        
        // Update confidence color
        const confidenceClass = this.getConfidenceClass(result.confidence)
        confidenceElement.className = `value ${confidenceClass}`
        confidenceFill.className = `confidence-fill ${confidenceClass}`
        
        // Update card styling based on waste type
        this.updateCardStyling(resultCard, result.class)
    }

    hideLiveResult() {
        document.getElementById('live-result').style.display = 'none'
    }

    showUploadedImage(file) {
        const img = document.getElementById('uploaded-image')
        const uploadArea = document.getElementById('upload-area')
        
        img.src = URL.createObjectURL(file)
        img.style.display = 'block'
        uploadArea.style.display = 'none'
    }

    showUploadResult(result) {
        document.getElementById('upload-result').style.display = 'block'
        this.updateUploadResult(result)
    }

    updateUploadResult(result) {
        const classElement = document.getElementById('upload-class')
        const confidenceElement = document.getElementById('upload-confidence')
        const confidenceFill = document.getElementById('upload-confidence-fill')
        const disposalInfo = document.getElementById('disposal-info')
        const resultCard = document.getElementById('upload-result')

        classElement.textContent = result.class
        confidenceElement.textContent = `${(result.confidence * 100).toFixed(1)}%`
        
        // Update confidence bar
        confidenceFill.style.width = `${result.confidence * 100}%`
        
        // Update confidence color
        const confidenceClass = this.getConfidenceClass(result.confidence)
        confidenceElement.className = `value ${confidenceClass}`
        confidenceFill.className = `confidence-fill ${confidenceClass}`
        
        // Update disposal information
        if (this.disposalInfo[result.class]) {
            const info = this.disposalInfo[result.class]
            disposalInfo.innerHTML = `
                <div class="disposal-message">
                    <span class="disposal-emoji">${info.emoji}</span>
                    <span>${info.message}</span>
                </div>
            `
            disposalInfo.style.display = 'block'
        }
        
        // Update card styling
        this.updateCardStyling(resultCard, result.class)
    }

    getConfidenceClass(confidence) {
        if (confidence >= 0.7) return 'high'
        if (confidence >= 0.5) return 'medium'
        return 'low'
    }

    updateCardStyling(card, wasteClass) {
        // Remove existing waste type classes
        card.classList.remove('organic-waste', 'hazardous-waste', 'inorganic-waste')
        
        // Add appropriate class
        if (wasteClass === 'Organic Waste') {
            card.classList.add('organic-waste')
        } else if (wasteClass === 'Hazardous Waste') {
            card.classList.add('hazardous-waste')
        } else if (wasteClass === 'Inorganic Waste') {
            card.classList.add('inorganic-waste')
        }
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay')
        overlay.style.opacity = '0'
        setTimeout(() => {
            overlay.style.display = 'none'
        }, 300)
    }
}