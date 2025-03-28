grep ggml_vec report.txt | awk '{print $6}' | cut -d'@' -f1
