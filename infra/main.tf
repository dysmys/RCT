terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

locals {
  region = join("-", slice(split("-", var.zone), 0, 2))
}

provider "google" {
  project = var.project
  region  = local.region
  zone    = var.zone
}

# ---------------------------------------------------------------------------
# Firewall — SSH access
# ---------------------------------------------------------------------------

resource "google_compute_firewall" "experiment_ssh" {
  name    = "experiment-allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["experiment-vm"]
}

# ---------------------------------------------------------------------------
# Persistent disk — repo clones for RCT runner
# ---------------------------------------------------------------------------

resource "google_compute_disk" "repo_store" {
  name = "experiment-repo-store"
  type = "pd-ssd"
  zone = var.zone
  size = var.repo_disk_size_gb

  labels = var.labels
}

# ---------------------------------------------------------------------------
# RCT experiment runner — consolidated Claude + Codex, high-parallelism
# n2-standard-32: 32 vCPU / 128 GB — supports 16+ parallel workers
# ---------------------------------------------------------------------------

resource "google_compute_instance" "rct_runner" {
  name         = "rct-runner"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["experiment-vm"]

  labels = merge(var.labels, { role = "rct-runner" })

  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
      size  = 100
      type  = "pd-ssd"
    }
  }

  attached_disk {
    source      = google_compute_disk.repo_store.self_link
    device_name = "repo-store"
    mode        = "READ_WRITE"
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = file("${path.module}/startup_experiment.sh")
}

# ---------------------------------------------------------------------------
# Belief preprocessing VM — extracts + embeds beliefs for all test.json repos
# ---------------------------------------------------------------------------

resource "google_compute_instance" "belief_preprocessor" {
  name         = "belief-preprocessor"
  machine_type = "e2-standard-4"
  zone         = var.zone
  tags         = ["experiment-vm"]

  labels = merge(var.labels, { role = "belief-preprocessor" })

  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
      size  = 50
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = file("${path.module}/startup_preprocess.sh")

  service_account {
    scopes = ["storage-rw", "logging-write"]
  }
}
