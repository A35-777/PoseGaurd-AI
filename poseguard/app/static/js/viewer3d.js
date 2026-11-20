/**
 * PoseGuard AI — Three.js 3D Skeletal Animation Viewport
 * 100% offline, lightweight, smooth skeletal bone visualization.
 */

class SkeletonViewer3D {
    constructor(canvasContainerId) {
        this.container = document.getElementById(canvasContainerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Skeletal objects
        this.jointMeshes = [];
        this.boneLines = null;
        this.gridHelper = null;
        this.groundPlane = null;
        this.trailLine = null;
        
        // Animation state
        this.jointNames = [];
        this.parents = [];
        this.frames = []; // [T, J, 3]
        this.totalFrames = 0;
        this.currentFrame = 0;
        this.fps = 30;
        this.groundY = 0.0;
        this.highlightDefect = false;
        
        this.init();
    }
    
    init() {
        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 500;
        
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0b0f19);
        this.scene.fog = new THREE.FogExp2(0x0b0f19, 0.08);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        this.camera.position.set(0, 1.4, 3.5);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);
        
        // Orbit Controls
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.target.set(0, 0.9, 0);
            this.controls.update();
        }
        
        // Lights
        const ambient = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambient);
        
        const dirLight = new THREE.DirectionalLight(0x06b6d4, 0.8);
        dirLight.position.set(3, 8, 4);
        this.scene.add(dirLight);
        
        // Ground Grid
        this.gridHelper = new THREE.GridHelper(10, 20, 0x06b6d4, 0x1f2937);
        this.gridHelper.position.y = 0.0;
        this.scene.add(this.gridHelper);
        
        // Window Resize
        window.addEventListener('resize', () => this.onResize());
        
        // Render Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }
    
    onResize() {
        if (!this.container || !this.renderer) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    setGroundPlane(gy) {
        this.groundY = gy;
        if (this.gridHelper) {
            this.gridHelper.position.y = gy;
        }
    }
    
    loadSkeleton(jointNames, parents, frames, fps = 30, groundY = 0.0) {
        this.jointNames = jointNames;
        this.parents = parents;
        this.frames = frames;
        this.totalFrames = frames.length;
        this.fps = fps;
        this.groundY = groundY;
        this.setGroundPlane(groundY);
        
        // Clear previous joints and bones
        this.jointMeshes.forEach(mesh => this.scene.remove(mesh));
        this.jointMeshes = [];
        if (this.boneLines) {
            this.scene.remove(this.boneLines);
            this.boneLines = null;
        }
        
        const numJoints = jointNames.length;
        const sphereGeo = new THREE.SphereGeometry(0.024, 16, 16);
        const jointMat = new THREE.MeshStandardMaterial({
            color: 0x06b6d4,
            roughness: 0.3,
            metalness: 0.6,
            emissive: 0x06b6d4,
            emissiveIntensity: 0.2
        });
        
        for (let i = 0; i < numJoints; i++) {
            const mesh = new THREE.Mesh(sphereGeo, jointMat.clone());
            this.scene.add(mesh);
            this.jointMeshes.push(mesh);
        }
        
        // Build lines for bones
        // Count bones (parents != -1)
        const linePositions = [];
        for (let i = 0; i < numJoints; i++) {
            if (this.parents[i] !== -1) {
                linePositions.push(0, 0, 0, 0, 0, 0); // 2 vertices per bone
            }
        }
        
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        const lineMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 2 });
        this.boneLines = new THREE.LineSegments(lineGeo, lineMat);
        this.scene.add(this.boneLines);
        
        this.updateFrame(0);
    }
    
    updateFrame(frameIdx, isDefectFrame = false) {
        if (!this.frames || this.frames.length === 0) return;
        this.currentFrame = Math.max(0, Math.min(frameIdx, this.totalFrames - 1));
        this.highlightDefect = isDefectFrame;
        
        const frameData = this.frames[this.currentFrame]; // [J, 3]
        if (!frameData) return;
        
        const numJoints = this.jointNames.length;
        const bonePositions = [];
        
        const normalColor = new THREE.Color(0x06b6d4);
        const defectColor = new THREE.Color(0xf43f5e); // Rose highlight on defect
        const activeColor = this.highlightDefect ? defectColor : normalColor;
        
        for (let j = 0; j < numJoints; j++) {
            const pos = frameData[j];
            if (!pos) continue;
            
            // In Three.js: X=right, Y=up, Z=depth
            const x = pos[0];
            const y = pos[1];
            const z = pos[2];
            
            const mesh = this.jointMeshes[j];
            if (mesh) {
                mesh.position.set(x, y, z);
                mesh.material.color.copy(activeColor);
                mesh.material.emissive.copy(activeColor);
            }
            
            const pIdx = this.parents[j];
            if (pIdx !== -1 && frameData[pIdx]) {
                const parentPos = frameData[pIdx];
                bonePositions.push(x, y, z);
                bonePositions.push(parentPos[0], parentPos[1], parentPos[2]);
            }
        }
        
        if (this.boneLines && bonePositions.length > 0) {
            this.boneLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(bonePositions, 3));
            this.boneLines.geometry.attributes.position.needsUpdate = true;
            this.boneLines.material.color.copy(this.highlightDefect ? defectColor : new THREE.Color(0x38bdf8));
        }
    }
    
    animate() {
        requestAnimationFrame(this.animate);
        if (this.controls) {
            this.controls.update();
        }
        this.renderer.render(this.scene, this.camera);
    }
}

