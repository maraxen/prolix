/**
 * Trajectory Viewer - py2Dmol Integration
 * 
 * Integrates py2Dmol's pseudo-3D viewer via iframe and postMessage communication.
 * This uses the actual py2Dmol JavaScript viewer for rendering.
 */

import type { TrajectoryData, Frame } from "./types";

export class TrajectoryViewer {
    private container: HTMLElement;
    private iframe: HTMLIFrameElement;

    private data: TrajectoryData | null = null;
    private currentFrame: number = 0;
    private isPlaying: boolean = false;
    private playInterval: number | null = null;
    private currentTrajectoryName: string = "Trajectory";

    constructor(containerId: string) {
        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`Container #${containerId} not found`);
        }
        this.container = container;

        // Create iframe to host py2Dmol viewer
        this.iframe = document.createElement("iframe");
        this.iframe.src = "/py2dmol_viewer.html";
        this.iframe.className = "viewer-iframe";
        this.iframe.style.width = "100%";
        this.iframe.style.height = "100%";
        this.iframe.style.border = "none";
        this.iframe.style.borderRadius = "12px";
        container.appendChild(this.iframe);

        // Listen for messages from iframe
        window.addEventListener("message", (e) => this.handleMessage(e));
    }

    private handleMessage(event: MessageEvent): void {
        // Handle messages from py2Dmol viewer if needed
        if (event.data?.type === "py2DmolReady") {
            console.log("py2Dmol viewer ready");
        }
    }

    private sendMessage(message: object): void {
        if (this.iframe.contentWindow) {
            this.iframe.contentWindow.postMessage(message, "*");
        }
    }

    async loadData(data: TrajectoryData): Promise<void> {
        this.data = data;
        this.currentFrame = 0;

        // Wait for iframe to be ready
        await this.waitForIframe();

        // Start a new trajectory
        this.sendMessage({
            type: "py2DmolNewTrajectory",
            name: this.currentTrajectoryName,
        });

        // Send first frame (or static structure if no trajectory)
        this.sendFrame(0);
    }

    private waitForIframe(): Promise<void> {
        return new Promise((resolve) => {
            if (this.iframe.contentDocument?.readyState === "complete") {
                // Give it a moment for JS to initialize
                setTimeout(resolve, 100);
            } else {
                this.iframe.onload = () => setTimeout(resolve, 100);
            }
        });
    }

    private sendFrame(frameIndex: number): void {
        if (!this.data) return;

        const atoms = this.data.atoms;
        let coords: [number, number, number][];

        if (this.data.trajectory && this.data.trajectory.length > 0) {
            coords = this.data.trajectory[frameIndex].coords;
        } else {
            coords = atoms.map(a => a.coord);
        }

        // Format data for py2Dmol
        const payload = {
            coords: coords,
            plddts: atoms.map(a => a.plddt ?? 70),
            chains: atoms.map(a => a.chain),
            atom_types: atoms.map(a => a.atomType ?? "P"),
        };

        this.sendMessage({
            type: "py2DmolUpdate",
            trajectoryName: this.currentTrajectoryName,
            payload,
        });
    }

    setFrame(frameIndex: number): void {
        if (!this.data) return;
        const maxFrame = this.data.trajectory?.length ?? 1;
        this.currentFrame = Math.max(0, Math.min(frameIndex, maxFrame - 1));
        this.sendFrame(this.currentFrame);
    }

    getFrame(): number {
        return this.currentFrame;
    }

    getFrameCount(): number {
        return this.data?.trajectory?.length ?? 1;
    }

    play(): void {
        if (this.isPlaying) return;
        this.isPlaying = true;

        this.playInterval = window.setInterval(() => {
            const maxFrame = this.getFrameCount();
            this.currentFrame = (this.currentFrame + 1) % maxFrame;
            this.sendFrame(this.currentFrame);

            // Dispatch event for controls to update
            window.dispatchEvent(new CustomEvent("framechange", {
                detail: { frame: this.currentFrame }
            }));
        }, 50); // 20 fps
    }

    pause(): void {
        this.isPlaying = false;
        if (this.playInterval !== null) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }
    }

    togglePlay(): boolean {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
        return this.isPlaying;
    }

    reset(): void {
        this.pause();
        this.currentFrame = 0;
        this.sendFrame(0);
    }

    setColorMode(mode: "chain" | "rainbow" | "plddt"): void {
        this.sendMessage({
            type: "py2DmolSetColorMode",
            mode,
        });
    }

    setShadow(enabled: boolean): void {
        this.sendMessage({
            type: "py2DmolSetShadow",
            enabled,
        });
    }

    setOutline(enabled: boolean): void {
        this.sendMessage({
            type: "py2DmolSetOutline",
            enabled,
        });
    }

    dispose(): void {
        this.pause();
        window.removeEventListener("message", (e) => this.handleMessage(e));
        this.iframe.remove();
    }
}
