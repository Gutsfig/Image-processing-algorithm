import os
import shutil

def batch_rename_paired_folders(vis_folder, ir_folder, start_index=1, dry_run=False):
    """
    批量重命名两个文件夹中成对的图像文件为序列格式（如 0001.png, 0002.png, ...）。

    参数:
    vis_folder (str): 可见光图像文件夹路径。
    ir_folder (str): 红外图像文件夹路径。
    start_index (int): 开始的序列号。
    dry_run (bool): 如果为 True，则只打印将要执行的操作，而不实际重命名文件。
    """
    print(f"--- Starting Batch Rename ---")
    print(f"Visible Light Folder: {vis_folder}")
    print(f"Infrared Folder:      {ir_folder}")
    if dry_run:
        print("\n*** DRY RUN MODE: No files will actually be renamed. ***\n")

    try:
        # 1. 获取并排序文件列表
        vis_files = sorted([f for f in os.listdir(vis_folder) if os.path.isfile(os.path.join(vis_folder, f))])
        ir_files = sorted([f for f in os.listdir(ir_folder) if os.path.isfile(os.path.join(ir_folder, f))])

        # 2. 检查文件数量是否一致
        if len(vis_files) != len(ir_files):
            print(f"Error: Mismatch in file count!")
            print(f"  - Visible folder has {len(vis_files)} files.")
            print(f"  - Infrared folder has {len(ir_files)} files.")
            print("Aborting.")
            return

        if not vis_files:
            print("No files found in the folders. Exiting.")
            return
            
        print(f"Found {len(vis_files)} pairs of files. Renaming...\n")
        
        # 3. 逐一重命名
        renamed_count = 0
        for i, (vis_filename, ir_filename) in enumerate(zip(vis_files, ir_files)):
            # 获取文件扩展名
            vis_ext = os.path.splitext(vis_filename)[1]
            ir_ext = os.path.splitext(ir_filename)[1]

            # 创建新的序列化文件名 (例如：0001.png)
            # 使用 zfill(4) 保证数字是4位数，不足的前面补0，如 1 -> "0001"
            new_base_name = str(start_index + i).zfill(4)
            new_vis_filename = f"{new_base_name}{vis_ext}"
            new_ir_filename = f"{new_base_name}{ir_ext}"

            # 定义原始和目标完整路径
            old_vis_path = os.path.join(vis_folder, vis_filename)
            new_vis_path = os.path.join(vis_folder, new_vis_filename)
            
            old_ir_path = os.path.join(ir_folder, ir_filename)
            new_ir_path = os.path.join(ir_folder, new_ir_filename)
            
            print(f"Pair {i+1}:")
            print(f"  vis: '{vis_filename}'  ->  '{new_vis_filename}'")
            print(f"  ir:  '{ir_filename}'  ->  '{new_ir_filename}'")

            if not dry_run:
                # 使用 shutil.move 来重命名，更安全
                try:
                    shutil.move(old_vis_path, new_vis_path)
                    shutil.move(old_ir_path, new_ir_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"  !! ERROR renaming this pair: {e}")
        
        print("\n--- Rename Complete ---")
        if not dry_run:
            print(f"Successfully renamed {renamed_count} pairs.")
        else:
            print("\n*** This was a DRY RUN. To apply changes, set dry_run=False. ***")


    except FileNotFoundError:
        print("Error: One of the specified folders does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- 配置区 ---
    # 定义你的文件夹路径
    base_test_folder = 'test'
    vis_image_folder = os.path.join(base_test_folder, 'vis')
    ir_image_folder = os.path.join(base_test_folder, 'ir')
    
    # 强烈建议先进行一次 "Dry Run" (试运行)
    # 这将只显示重命名计划，而不会实际修改任何文件。
    # 确认计划无误后，再将 dry_run 设置为 False。
    IS_DRY_RUN = False
    
    # 调用函数
    batch_rename_paired_folders(vis_image_folder, ir_image_folder, dry_run=IS_DRY_RUN)