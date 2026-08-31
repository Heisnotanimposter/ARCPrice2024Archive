python main.py --train_data_path '/Users/seungwonlee/ARC_Prize_2024/ARCPrize2024/arc-prize-2024-local/arc-agi_evaluation_challenges.json' \
               --test_data_path '/Users/seungwonlee/ARC_Prize_2024/ARCPrize2024/arc-prize-2024-local/arc-agi_evaluation_solutions.json' \
               --target_directory '/Users/seungwonlee/ARC_Prize_2024/ARCPrize2024/arc-prize-2024-colab2' \
               --num_epochs 50 \
               --batch_size 128 \
               --learning_rate 0.0005 \
               --hidden_size 256 \
               --num_layers 3 \
               --dropout_rate 0.2