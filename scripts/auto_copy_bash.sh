self_path=$0
output_dir=scripts/test
output_file="$output_dir/$(basename $self_path)"
cp ${self_path} ${output_file}

if [[ -f "$output_file" ]]; then
    echo "Finetune Script copied successfully to $output_file"
else
    echo "Failed to copy the script."
fi