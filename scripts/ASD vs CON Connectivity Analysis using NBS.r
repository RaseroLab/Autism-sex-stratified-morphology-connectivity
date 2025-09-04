# ASD vs CON Connectivity Analysis using NBS
# Description: Standalone pipeline for NBS-based connectivity analysis with covariates (no interaction terms)

# ---------------------------
# Load Required Packages
# ---------------------------
packages <- c("tidyverse", "NBS", "RColorBrewer", "reshape2")
invisible(lapply(packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg, dependencies = TRUE)
  library(pkg, character.only = TRUE)
}))

# ---------------------------
# User Inputs (Edit Here)
# ---------------------------
conn_dir <- "data/conn_matrices/"         # Folder of 68x68 matrices (RDS or CSV per subject)
cov_file <- "data/covariates.csv"         # Subject-level covariates
lut_file <- "data/lookup_table.txt"       # ROI label file
out_dir  <- "results/"                    # Output folder

# ---------------------------
# Create Output Folders
# ---------------------------
dir.create(out_dir, showWarnings = FALSE)
dir.create(file.path(out_dir, "figures"), showWarnings = FALSE)
dir.create(file.path(out_dir, "stats"), showWarnings = FALSE)

# ---------------------------
# Load Data
# ---------------------------
cat("Loading connectivity matrices...\n")
conn_files <- list.files(conn_dir, pattern = "\\.rds$|\\.csv$", full.names = TRUE)
conn_list <- lapply(conn_files, function(f) {
  mat <- if (grepl("\\.csv$", f)) as.matrix(read.csv(f, row.names = 1)) else readRDS(f)
  stopifnot(nrow(mat) == 68, ncol(mat) == 68)
  mat
})
names(conn_list) <- gsub("\\..*$", "", basename(conn_files))

cat("Loading covariates...\n")
cov <- read.csv(cov_file)
cov$ID <- as.character(cov$ID)
conn_subjects <- intersect(names(conn_list), cov$ID)

if (length(conn_subjects) < length(conn_list)) {
  warning("Some matrices or covariates are unmatched and will be dropped.")
}
conn_list <- conn_list[conn_subjects]
cov <- cov %>% filter(ID %in% conn_subjects)

# ---------------------------
# Build Design Matrix (Main Effects Only)
# ---------------------------
design <- model.matrix(~ Group + Age + TIV + Site, data = cov)

# ---------------------------
# Flatten Matrices
# ---------------------------
cat("Flattening connectivity data...\n")
lower_tri_idx <- lower.tri(conn_list[[1]])
X <- do.call(rbind, lapply(conn_list, function(m) m[lower_tri_idx]))

# ---------------------------
# Run NBS Analysis
# ---------------------------
cat("Running NBS permutation test...\n")
set.seed(2025)

# Determine contrast vector length dynamically
colnames(design)
# Example: if Group is the second column in the design matrix
contrast_vector <- rep(0, ncol(design))
contrast_vector[2] <- 1  # Assuming Group is the second column

nbr_out <- NBS::nbr_analysis(
  conn_mat = X,
  design = design,
  contrast = contrast_vector,
  n_perm = 1000,
  p_thresh = 0.05
)

saveRDS(nbr_out, file.path(out_dir, "stats", "nbr_result.rds"))
write.csv(nbr_out$significant_edges, file.path(out_dir, "stats", "sig_edges.csv"), row.names = FALSE)

# ---------------------------
# Plot Significant Edge Mask
# ---------------------------
cat("Plotting results...\n")
sig_mask <- matrix(0, 68, 68)
edges <- nbr_out$significant_edges
for (i in 1:nrow(edges)) {
  a <- edges[i, "node1"]
  b <- edges[i, "node2"]
  sig_mask[a, b] <- 1
  sig_mask[b, a] <- 1
}

png(file.path(out_dir, "figures", "sig_edge_mask.png"), width = 800, height = 800)
heatmap(sig_mask, Rowv = NA, Colv = NA,
        col = c("white", "red"),
        main = "Significant NBS Edge Mask", xlab = "ROI", ylab = "ROI")
dev.off()

# ---------------------------
# Save Session Info
# ---------------------------
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))

cat("Pipeline complete. Results saved in:", out_dir, "\n")
