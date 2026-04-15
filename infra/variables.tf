variable "project" {
  description = "GCP project ID"
  type        = string
}

variable "zone" {
  description = "GCP zone"
  type        = string
}

variable "machine_type" {
  description = "Machine type for the RCT runner VM"
  type        = string
  default     = "n2-standard-32"
}

variable "repo_disk_size_gb" {
  description = "Size of persistent SSD disk for repo clones (GB)"
  type        = number
  default     = 500
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default     = {}
}
