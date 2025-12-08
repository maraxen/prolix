/**
 * Prolix Trajectory Viewer - Main Application
 * 
 * A browser-based viewer for molecular dynamics trajectories from prolix.
 * Supports loading array_record trajectory files and PDB structures.
 */

import "./index.css";
import { TrajectoryViewer } from "./viewer";
import { FileLoader } from "./fileLoader";
import { Controls } from "./controls";

// Initialize the application when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  const app = new App();
  app.init();
});

class App {
  private viewer: TrajectoryViewer | null = null;
  private fileLoader: FileLoader;
  private controls: Controls | null = null;

  constructor() {
    this.fileLoader = new FileLoader();
  }

  init(): void {
    this.renderUI();
    this.setupEventListeners();
    this.checkServerConfig();
  }

  private async checkServerConfig(): Promise<void> {
    try {
      const response = await fetch("/api/config");
      if (response.ok) {
        const config = await response.json();
        if (config.pdbPath) {
          console.log("Found server config, auto-loading...", config);
          this.loadFromApi();
        }
      }
    } catch (e) {
      console.log("No API server detected or config fetch failed", e);
    }
  }

  private async loadFromApi(): Promise<void> {
    this.showLoading("Loading data from server...");
    try {
      const data = await this.fileLoader.loadFromUrl("/api/pdb", "/api/trajectory");
      this.showViewerSection();

      this.viewer = new TrajectoryViewer("py2dmol-container");
      await this.viewer.loadData(data);

      this.controls = new Controls(this.viewer, data.trajectory?.length ?? 1);
      this.hideLoading();
    } catch (error) {
      this.hideLoading();
      console.error(error);
      alert(`Error auto-loading from server: ${error}`);
    }
  }

  private renderUI(): void {
    const root = document.getElementById("root");
    if (!root) return;

    root.innerHTML = `
      <div class="app-container">
        <header class="app-header">
          <h1>Prolix Trajectory Viewer</h1>
          <p class="subtitle">Visualize molecular dynamics simulations</p>
        </header>

        <main class="app-main">
          <section class="file-section" id="file-section">
            <div class="file-upload-area" id="drop-zone">
              <div class="upload-icon">📁</div>
              <h2>Load Trajectory</h2>
              <p>Drag & drop files here, or click to select</p>
              
              <div class="file-inputs">
                <div class="file-input-group">
                  <label for="pdb-file">PDB Structure (required)</label>
                  <input type="file" id="pdb-file" accept=".pdb,.cif" />
                  <span class="file-name" id="pdb-file-name">No file selected</span>
                </div>
                
                <div class="file-input-group">
                  <label for="trajectory-file">Trajectory (optional)</label>
                  <input type="file" id="trajectory-file" accept=".array_record,.msgpack,.json" />
                  <span class="file-name" id="trajectory-file-name">No file selected</span>
                </div>
              </div>

              <button class="load-button" id="load-button" disabled>
                Load Structure
              </button>
            </div>
          </section>

          <section class="viewer-section" id="viewer-section" style="display: none;">
            <div class="viewer-container">
              <canvas id="viewer-canvas"></canvas>
              <div id="py2dmol-container"></div>
            </div>
            
            <div class="controls-panel" id="controls-panel">
              <div class="control-group">
                <label>Frame</label>
                <input type="range" id="frame-slider" min="0" max="100" value="0" />
                <span id="frame-counter">0 / 0</span>
              </div>
              
              <div class="control-group">
                <label>Playback</label>
                <button id="play-button">▶ Play</button>
                <button id="reset-button">⏮ Reset</button>
              </div>

              <div class="control-group">
                <label>Style</label>
                <select id="color-mode">
                  <option value="chain">By Chain</option>
                  <option value="rainbow">Rainbow</option>
                  <option value="plddt">By pLDDT</option>
                </select>
              </div>

              <div class="control-group">
                <label>Options</label>
                <label class="checkbox-label">
                  <input type="checkbox" id="show-shadow" checked />
                  Shadow
                </label>
                <label class="checkbox-label">
                  <input type="checkbox" id="show-outline" checked />
                  Outline
                </label>
              </div>

              <button class="back-button" id="back-button">← Load New File</button>
            </div>
          </section>
        </main>

        <footer class="app-footer">
          <p>Powered by <a href="https://github.com/sokrypton/py2Dmol" target="_blank">py2Dmol</a></p>
        </footer>
      </div>

      <div class="loading-overlay" id="loading-overlay" style="display: none;">
        <div class="loading-spinner"></div>
        <p id="loading-message">Loading...</p>
      </div>
    `;
  }

  private setupEventListeners(): void {
    // File inputs
    const pdbInput = document.getElementById("pdb-file") as HTMLInputElement;
    const trajectoryInput = document.getElementById("trajectory-file") as HTMLInputElement;
    const loadButton = document.getElementById("load-button") as HTMLButtonElement;
    const backButton = document.getElementById("back-button") as HTMLButtonElement;

    pdbInput?.addEventListener("change", () => this.onFileSelected("pdb", pdbInput));
    trajectoryInput?.addEventListener("change", () => this.onFileSelected("trajectory", trajectoryInput));
    loadButton?.addEventListener("click", () => this.loadFiles());
    backButton?.addEventListener("click", () => this.showFileSection());

    // Drag and drop
    const dropZone = document.getElementById("drop-zone");
    if (dropZone) {
      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
      dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
      });
      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        this.handleDrop(e);
      });
    }
  }

  private onFileSelected(type: "pdb" | "trajectory", input: HTMLInputElement): void {
    const file = input.files?.[0];
    const nameSpan = document.getElementById(`${type === "pdb" ? "pdb" : "trajectory"}-file-name`);

    if (file && nameSpan) {
      nameSpan.textContent = file.name;
      this.fileLoader.setFile(type, file);
    }

    this.updateLoadButton();
  }

  private handleDrop(e: DragEvent): void {
    const files = e.dataTransfer?.files;
    if (!files) return;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.name.endsWith(".pdb") || file.name.endsWith(".cif")) {
        this.fileLoader.setFile("pdb", file);
        const nameSpan = document.getElementById("pdb-file-name");
        if (nameSpan) nameSpan.textContent = file.name;
      } else if (file.name.endsWith(".array_record") || file.name.endsWith(".msgpack") || file.name.endsWith(".json")) {
        this.fileLoader.setFile("trajectory", file);
        const nameSpan = document.getElementById("trajectory-file-name");
        if (nameSpan) nameSpan.textContent = file.name;
      }
    }

    this.updateLoadButton();
  }

  private updateLoadButton(): void {
    const loadButton = document.getElementById("load-button") as HTMLButtonElement;
    if (loadButton) {
      loadButton.disabled = !this.fileLoader.hasPdbFile();
    }
  }

  private async loadFiles(): Promise<void> {
    this.showLoading("Loading structure...");

    try {
      const data = await this.fileLoader.load();

      this.showViewerSection();

      // Initialize viewer with loaded data
      this.viewer = new TrajectoryViewer("py2dmol-container");
      await this.viewer.loadData(data);

      // Initialize controls
      this.controls = new Controls(this.viewer, data.trajectory?.length ?? 1);

      this.hideLoading();
    } catch (error) {
      this.hideLoading();
      alert(`Error loading files: ${error}`);
    }
  }

  private showLoading(message: string): void {
    const overlay = document.getElementById("loading-overlay");
    const messageEl = document.getElementById("loading-message");
    if (overlay) overlay.style.display = "flex";
    if (messageEl) messageEl.textContent = message;
  }

  private hideLoading(): void {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.style.display = "none";
  }

  private showViewerSection(): void {
    const fileSection = document.getElementById("file-section");
    const viewerSection = document.getElementById("viewer-section");
    if (fileSection) fileSection.style.display = "none";
    if (viewerSection) viewerSection.style.display = "block";
  }

  private showFileSection(): void {
    const fileSection = document.getElementById("file-section");
    const viewerSection = document.getElementById("viewer-section");
    if (fileSection) fileSection.style.display = "block";
    if (viewerSection) viewerSection.style.display = "none";

    // Clean up viewer
    if (this.viewer) {
      this.viewer.dispose();
      this.viewer = null;
    }
    if (this.controls) {
      this.controls.dispose();
      this.controls = null;
    }
  }
}
