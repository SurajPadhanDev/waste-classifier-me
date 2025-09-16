export class CameraManager {
    constructor() {
        this.stream = null
        this.video = null
        this.canvas = null
        this.context = null
        this.active = false
    }

    async start() {
        try {
            // Get video and canvas elements
            this.video = document.getElementById('camera-feed')
            this.canvas = document.getElementById('camera-canvas')
            this.context = this.canvas.getContext('2d')

            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'environment' // Use back camera on mobile
                },
                audio: false
            })

            // Set video source
            this.video.srcObject = this.stream
            this.active = true

            // Wait for video to be ready
            return new Promise((resolve, reject) => {
                this.video.onloadedmetadata = () => {
                    this.video.play()
                    
                    // Set canvas dimensions to match video
                    this.canvas.width = this.video.videoWidth
                    this.canvas.height = this.video.videoHeight
                    
                    resolve(this.stream)
                }
                
                this.video.onerror = reject
            })

        } catch (error) {
            console.error('Camera access error:', error)
            this.active = false
            throw error
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop())
            this.stream = null
        }
        
        if (this.video) {
            this.video.srcObject = null
        }
        
        this.active = false
    }

    isActive() {
        return this.active && this.video && this.video.readyState === 4
    }

    captureFrame() {
        if (!this.isActive()) {
            return null
        }

        try {
            // Draw current video frame to canvas
            this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height)
            
            // Get image data from canvas
            const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height)
            
            return imageData
            
        } catch (error) {
            console.error('Frame capture error:', error)
            return null
        }
    }

    // Get current frame as blob for upload
    async getFrameBlob() {
        if (!this.isActive()) {
            return null
        }

        return new Promise((resolve) => {
            this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height)
            this.canvas.toBlob(resolve, 'image/jpeg', 0.8)
        })
    }
}