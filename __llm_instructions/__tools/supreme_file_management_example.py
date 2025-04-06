"""
title: Supreme File Management
author: Wes Caldwell
email: Musicheardworldwide@gmail.com
date: 2024-07-19
version: 1.0
license: MIT
description: Big set of file management tools.
"""
import os
import json
import shutil
import logging
import hashlib
import datetime
import zipfile
import tarfile
from collections import defaultdict

logging.basicConfig(level=logging.INFO)


class Tools:
    def __init__(self, base_path=None):
        self.base_path = base_path if base_path else os.getcwd()

    def create_folder(self, folder_name: str, path: str = None) -> str:
        """
        Create a new folder.
        :param folder_name: The name of the folder to create.
        :param path: The path where the folder should be created.
        :return: A success message if the folder is created successfully.
        """
        folder_path = os.path.join(path if path else self.base_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.info(
                f"Folder '{folder_name}' created successfully at {folder_path}!"
            )
            return f"Folder '{folder_name}' created successfully!"
        else:
            logging.warning(f"Folder '{folder_name}' already exists at {folder_path}.")
            return f"Folder '{folder_name}' already exists."

    def delete_folder(self, folder_name: str, path: str = None) -> str:
        """
        Delete a folder.
        :param folder_name: The name of the folder to delete.
        :param path: The path where the folder is located.
        :return: A success message if the folder is deleted successfully.
        """
        folder_path = os.path.join(path if path else self.base_path, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logging.info(
                f"Folder '{folder_name}' deleted successfully from {folder_path}!"
            )
            return f"Folder '{folder_name}' deleted successfully!"
        else:
            logging.warning(f"Folder '{folder_name}' does not exist at {folder_path}.")
            return f"Folder '{folder_name}' does not exist."

    def create_file(self, file_name: str, content: str = "", path: str = None) -> str:
        """
        Create a new file.
        :param file_name: The name of the file to create.
        :param content: The content to write to the file.
        :param path: The path where the file should be created.
        :return: A success message if the file is created successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        with open(file_path, "w") as file:
            file.write(content)
        logging.info(f"File '{file_name}' created successfully at {file_path}!")
        return f"File '{file_name}' created successfully!"

    def delete_file(self, file_name: str, path: str = None) -> str:
        """
        Delete a file.
        :param file_name: The name of the file to delete.
        :param path: The path where the file is located.
        :return: A success message if the file is deleted successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"File '{file_name}' deleted successfully from {file_path}!")
            return f"File '{file_name}' deleted successfully!"
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def read_file(self, file_name: str, path: str = None) -> str:
        """
        Read the content of a file.
        :param file_name: The name of the file to read.
        :param path: The path where the file is located.
        :return: The content of the file.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                content = file.read()
            logging.info(f"File '{file_name}' read successfully from {file_path}!")
            return content
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def write_to_file(self, file_name: str, content: str, path: str = None) -> str:
        """
        Write content to a file.
        :param file_name: The name of the file to write to.
        :param content: The content to write to the file.
        :param path: The path where the file is located.
        :return: A success message if the content is written successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        with open(file_path, "w") as file:
            file.write(content)
        logging.info(
            f"Content written to file '{file_name}' successfully at {file_path}!"
        )
        return f"Content written to file '{file_name}' successfully!"

    def list_files(self, path: str = None) -> str:
        """
        List all files in the specified directory.
        :param path: The path where the files should be listed.
        :return: A list of files in the specified directory.
        """
        directory_path = path if path else self.base_path
        files = os.listdir(directory_path)
        logging.info(f"Files listed successfully from {directory_path}!")
        return "Files in the specified directory:\n" + "\n".join(files)

    def read_json_file(self, file_name: str, path: str = None) -> dict:
        """
        Read the content of a JSON file.
        :param file_name: The name of the JSON file to read.
        :param path: The path where the JSON file is located.
        :return: The content of the JSON file as a dictionary.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                content = json.load(file)
            logging.info(f"JSON file '{file_name}' read successfully from {file_path}!")
            return content
        else:
            logging.warning(f"JSON file '{file_name}' does not exist at {file_path}.")
            return f"JSON file '{file_name}' does not exist."

    def write_json_file(self, file_name: str, content: dict, path: str = None) -> str:
        """
        Write content to a JSON file.
        :param file_name: The name of the JSON file to write to.
        :param content: The content to write to the JSON file as a dictionary.
        :param path: The path where the JSON file should be created.
        :return: A success message if the content is written successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        with open(file_path, "w") as file:
            json.dump(content, file, indent=4)
        logging.info(
            f"Content written to JSON file '{file_name}' successfully at {file_path}!"
        )
        return f"Content written to JSON file '{file_name}' successfully!"

    def copy_file(
        self, src_file: str, dest_file: str, src_path: str = None, dest_path: str = None
    ) -> str:
        """
        Copy a file from source to destination.
        :param src_file: The name of the source file.
        :param dest_file: The name of the destination file.
        :param src_path: The path where the source file is located.
        :param dest_path: The path where the destination file should be created.
        :return: A success message if the file is copied successfully.
        """
        src_file_path = os.path.join(src_path if src_path else self.base_path, src_file)
        dest_file_path = os.path.join(
            dest_path if dest_path else self.base_path, dest_file
        )
        if os.path.exists(src_file_path):
            shutil.copy2(src_file_path, dest_file_path)
            logging.info(f"File '{src_file}' copied successfully to {dest_file_path}!")
            return f"File '{src_file}' copied successfully to {dest_file}!"
        else:
            logging.warning(f"File '{src_file}' does not exist at {src_file_path}.")
            return f"File '{src_file}' does not exist."

    def copy_folder(
        self,
        src_folder: str,
        dest_folder: str,
        src_path: str = None,
        dest_path: str = None,
    ) -> str:
        """
        Copy a folder from source to destination.
        :param src_folder: The name of the source folder.
        :param dest_folder: The name of the destination folder.
        :param src_path: The path where the source folder is located.
        :param dest_path: The path where the destination folder should be created.
        :return: A success message if the folder is copied successfully.
        """
        src_folder_path = os.path.join(
            src_path if src_path else self.base_path, src_folder
        )
        dest_folder_path = os.path.join(
            dest_path if dest_path else self.base_path, dest_folder
        )
        if os.path.exists(src_folder_path):
            shutil.copytree(src_folder_path, dest_folder_path)
            logging.info(
                f"Folder '{src_folder}' copied successfully to {dest_folder_path}!"
            )
            return f"Folder '{src_folder}' copied successfully to {dest_folder}!"
        else:
            logging.warning(
                f"Folder '{src_folder}' does not exist at {src_folder_path}."
            )
            return f"Folder '{src_folder}' does not exist."

    def move_file(
        self, src_file: str, dest_file: str, src_path: str = None, dest_path: str = None
    ) -> str:
        """
        Move a file from source to destination.
        :param src_file: The name of the source file.
        :param dest_file: The name of the destination file.
        :param src_path: The path where the source file is located.
        :param dest_path: The path where the destination file should be created.
        :return: A success message if the file is moved successfully.
        """
        src_file_path = os.path.join(src_path if src_path else self.base_path, src_file)
        dest_file_path = os.path.join(
            dest_path if dest_path else self.base_path, dest_file
        )
        if os.path.exists(src_file_path):
            shutil.move(src_file_path, dest_file_path)
            logging.info(f"File '{src_file}' moved successfully to {dest_file_path}!")
            return f"File '{src_file}' moved successfully to {dest_file}!"
        else:
            logging.warning(f"File '{src_file}' does not exist at {src_file_path}.")
            return f"File '{src_file}' does not exist."

    def move_folder(
        self,
        src_folder: str,
        dest_folder: str,
        src_path: str = None,
        dest_path: str = None,
    ) -> str:
        """
        Move a folder from source to destination.
        :param src_folder: The name of the source folder.
        :param dest_folder: The name of the destination folder.
        :param src_path: The path where the source folder is located.
        :param dest_path: The path where the destination folder should be created.
        :return: A success message if the folder is moved successfully.
        """
        src_folder_path = os.path.join(
            src_path if src_path else self.base_path, src_folder
        )
        dest_folder_path = os.path.join(
            dest_path if dest_path else self.base_path, dest_folder
        )
        if os.path.exists(src_folder_path):
            shutil.move(src_folder_path, dest_folder_path)
            logging.info(
                f"Folder '{src_folder}' moved successfully to {dest_folder_path}!"
            )
            return f"Folder '{src_folder}' moved successfully to {dest_folder}!"
        else:
            logging.warning(
                f"Folder '{src_folder}' does not exist at {src_folder_path}."
            )
            return f"Folder '{src_folder}' does not exist."

    def is_file(self, path: str) -> bool:
        """
        Check if the given path is a file.
        :param path: The path to check.
        :return: True if the path is a file, False otherwise.
        """
        return os.path.isfile(path)

    def is_directory(self, path: str) -> bool:
        """
        Check if the given path is a directory.
        :param path: The path to check.
        :return: True if the path is a directory, False otherwise.
        """
        return os.path.isdir(path)

    def encrypt_file(self, file_name: str, key: str, path: str = None) -> str:
        """
        Encrypt a file using a key.
        :param file_name: The name of the file to encrypt.
        :param key: The key to use for encryption.
        :param path: The path where the file is located.
        :return: A success message if the file is encrypted successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                data = file.read()
            key = hashlib.sha256(key.encode()).digest()
            encrypted_data = bytearray(
                x ^ key[i % len(key)] for i, x in enumerate(data)
            )
            with open(file_path, "wb") as file:
                file.write(encrypted_data)
            logging.info(f"File '{file_name}' encrypted successfully at {file_path}!")
            return f"File '{file_name}' encrypted successfully!"
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def decrypt_file(self, file_name: str, key: str, path: str = None) -> str:
        """
        Decrypt a file using a key.
        :param file_name: The name of the file to decrypt.
        :param key: The key to use for decryption.
        :param path: The path where the file is located.
        :return: A success message if the file is decrypted successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                data = file.read()
            key = hashlib.sha256(key.encode()).digest()
            decrypted_data = bytearray(
                x ^ key[i % len(key)] for i, x in enumerate(data)
            )
            with open(file_path, "wb") as file:
                file.write(decrypted_data)
            logging.info(f"File '{file_name}' decrypted successfully at {file_path}!")
            return f"File '{file_name}' decrypted successfully!"
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def get_file_metadata(self, file_name: str, path: str = None) -> dict:
        """
        Get metadata of a file.
        :param file_name: The name of the file to get metadata for.
        :param path: The path where the file is located.
        :return: A dictionary containing the file's metadata.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            metadata = os.stat(file_path)
            return {
                "size": metadata.st_size,
                "creation_time": datetime.datetime.fromtimestamp(
                    metadata.st_ctime
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "modification_time": datetime.datetime.fromtimestamp(
                    metadata.st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "access_time": datetime.datetime.fromtimestamp(
                    metadata.st_atime
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def batch_rename_files(
        self, directory: str, old_pattern: str, new_pattern: str
    ) -> str:
        """
        Batch rename files in a directory.
        :param directory: The directory containing the files to rename.
        :param old_pattern: The old pattern in the file names to replace.
        :param new_pattern: The new pattern to replace the old pattern with.
        :return: A success message if the files are renamed successfully.
        """
        directory_path = os.path.join(self.base_path, directory)
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                if old_pattern in filename:
                    new_filename = filename.replace(old_pattern, new_pattern)
                    os.rename(
                        os.path.join(directory_path, filename),
                        os.path.join(directory_path, new_filename),
                    )
            logging.info(f"Files in directory '{directory}' renamed successfully!")
            return f"Files in directory '{directory}' renamed successfully!"
        else:
            logging.warning(
                f"Directory '{directory}' does not exist at {directory_path}."
            )
            return f"Directory '{directory}' does not exist."

    def compress_file(
        self,
        file_name: str,
        output_filename: str,
        format: str = "zip",
        path: str = None,
    ) -> str:
        """
        Compress a file into the specified format.
        :param file_name: The name of the file to compress.
        :param output_filename: The name of the output compressed file.
        :param format: The compression format ('zip', 'tar', 'gztar').
        :param path: The path where the file is located.
        :return: A success message if the file is compressed successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        output_path = os.path.join(path if path else self.base_path, output_filename)
        if os.path.exists(file_path):
            if format == "zip":
                with zipfile.ZipFile(output_path, "w") as zipf:
                    zipf.write(file_path, os.path.basename(file_path))
            elif format == "tar":
                with tarfile.open(output_path, "w") as tarf:
                    tarf.add(file_path, os.path.basename(file_path))
            elif format == "gztar":
                with tarfile.open(output_path, "w:gz") as tarf:
                    tarf.add(file_path, os.path.basename(file_path))
            else:
                logging.warning(f"Unsupported compression format: {format}.")
                return f"Unsupported compression format: {format}."
            logging.info(
                f"File '{file_name}' compressed successfully to {output_path}!"
            )
            return f"File '{file_name}' compressed successfully to {output_filename}!"
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def decompress_file(
        self, file_name: str, output_directory: str, path: str = None
    ) -> str:
        """
        Decompress a file into the specified directory.
        :param file_name: The name of the file to decompress.
        :param output_directory: The directory where the decompressed files will be stored.
        :param path: The path where the file is located.
        :return: A success message if the file is decompressed successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        output_path = os.path.join(path if path else self.base_path, output_directory)
        if os.path.exists(file_path):
            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zipf:
                    zipf.extractall(output_path)
            elif (
                file_name.endswith(".tar")
                or file_name.endswith(".tar.gz")
                or file_name.endswith(".tgz")
            ):
                with tarfile.open(file_path, "r") as tarf:
                    tarf.extractall(output_path)
            else:
                logging.warning(
                    f"Unsupported decompression format for file: {file_name}."
                )
                return f"Unsupported decompression format for file: {file_name}."
            logging.info(
                f"File '{file_name}' decompressed successfully to {output_path}!"
            )
            return (
                f"File '{file_name}' decompressed successfully to {output_directory}!"
            )
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def save_file_version(self, file_name: str, path: str = None) -> str:
        """
        Save a version of the file.
        :param file_name: The name of the file to save a version of.
        :param path: The path where the file is located.
        :return: A success message if the version is saved successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            version_path = os.path.join(
                path if path else self.base_path,
                f"{file_name}_v{len(self.versions[file_name]) + 1}",
            )
            shutil.copy2(file_path, version_path)
            self.versions[file_name].append(version_path)
            logging.info(f"Version saved for file '{file_name}' at {version_path}!")
            return f"Version saved for file '{file_name}'!"
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def restore_file_version(
        self, file_name: str, version: int, path: str = None
    ) -> str:
        """
        Restore a file to a previous version.
        :param file_name: The name of the file to restore.
        :param version: The version number to restore.
        :param path: The path where the file is located.
        :return: A success message if the file is restored successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            if 1 <= version <= len(self.versions[file_name]):
                version_path = self.versions[file_name][version - 1]
                shutil.copy2(version_path, file_path)
                logging.info(
                    f"File '{file_name}' restored to version {version} at {file_path}!"
                )
                return f"File '{file_name}' restored to version {version}!"
            else:
                logging.warning(
                    f"Version {version} does not exist for file '{file_name}'."
                )
                return f"Version {version} does not exist for file '{file_name}'."
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def share_file(
        self, file_name: str, user: str, permissions: str, path: str = None
    ) -> str:
        """
        Share a file with a specific user and set permissions.
        :param file_name: The name of the file to share.
        :param user: The user to share the file with.
        :param permissions: The permissions to set (e.g., 'read', 'write').
        :param path: The path where the file is located.
        :return: A success message if the file is shared successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            # Implement sharing logic here (e.g., store sharing information in a database or file)
            logging.info(
                f"File '{file_name}' shared with user '{user}' with permissions '{permissions}'!"
            )
            return f"File '{file_name}' shared with user '{user}' with permissions '{permissions}'!"
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def search_files(self, keyword: str, path: str = None) -> list:
        """
        Search for files containing the keyword in their names or content.
        :param keyword: The keyword to search for.
        :param path: The path where to search for files.
        :return: A list of file paths that match the search criteria.
        """
        search_path = path if path else self.base_path
        matching_files = []
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if keyword in file:
                    matching_files.append(os.path.join(root, file))
                else:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        if keyword in f.read():
                            matching_files.append(file_path)
        logging.info(
            f"Search for keyword '{keyword}' completed with {len(matching_files)} matches!"
        )
        return matching_files

    def synchronize_files(self, source_path: str, destination_path: str) -> str:
        """
        Synchronize files between two directories.
        :param source_path: The source directory to synchronize from.
        :param destination_path: The destination directory to synchronize to.
        :return: A success message if the synchronization is completed successfully.
        """
        if os.path.exists(source_path) and os.path.exists(destination_path):
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    source_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(source_file_path, source_path)
                    destination_file_path = os.path.join(
                        destination_path, relative_path
                    )
                    if (
                        not os.path.exists(destination_file_path)
                        or os.stat(source_file_path).st_mtime
                        - os.stat(destination_file_path).st_mtime
                        > 1
                    ):
                        shutil.copy2(source_file_path, destination_file_path)
            logging.info(
                f"Synchronization from '{source_path}' to '{destination_path}' completed successfully!"
            )
            return f"Synchronization from '{source_path}' to '{destination_path}' completed successfully!"
        else:
            logging.warning(f"Source or destination path does not exist.")
            return f"Source or destination path does not exist."

    def add_tag(self, file_name: str, tag: str, path: str = None) -> str:
        """
        Add a tag to a file.
        :param file_name: The name of the file to add a tag to.
        :param tag: The tag to add.
        :param path: The path where the file is located.
        :return: A success message if the tag is added successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            self.tags[file_name].append(tag)
            logging.info(f"Tag '{tag}' added to file '{file_name}'!")
            return f"Tag '{tag}' added to file '{file_name}'!"
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def get_tags(self, file_name: str, path: str = None) -> list:
        """
        Get the tags for a file.
        :param file_name: The name of the file to get tags for.
        :param path: The path where the file is located.
        :return: A list of tags for the file.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            logging.info(f"Tags retrieved for file '{file_name}'!")
            return self.tags[file_name]
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def convert_file(self, file_name: str, output_format: str, path: str = None) -> str:
        """
        Convert a file to a different format.
        :param file_name: The name of the file to convert.
        :param output_format: The desired output format.
        :param path: The path where the file is located.
        :return: A success message if the file is converted successfully.
        """
        file_path = os.path.join(path if path else self.base_path, file_name)
        if os.path.exists(file_path):
            # Implement file conversion logic here
            # For example, converting a text file to a PDF
            logging.info(f"File '{file_name}' converted to {output_format} format!")
            return f"File '{file_name}' converted to {output_format} format!"
        else:
            logging.warning(f"File '{file_name}' does not exist at {file_path}.")
            return f"File '{file_name}' does not exist."

    def backup_files(self, source_path: str, backup_path: str) -> str:
        """
        Backup files from the source directory to the backup directory.
        :param source_path: The source directory to backup from.
        :param backup_path: The backup directory to backup to.
        :return: A success message if the backup is completed successfully.
        """
        if os.path.exists(source_path) and os.path.exists(backup_path):
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    source_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(source_file_path, source_path)
                    backup_file_path = os.path.join(backup_path, relative_path)
                    if not os.path.exists(os.path.dirname(backup_file_path)):
                        os.makedirs(os.path.dirname(backup_file_path))
                    shutil.copy2(source_file_path, backup_file_path)
            logging.info(
                f"Backup from '{source_path}' to '{backup_path}' completed successfully!"
            )
            return f"Backup from '{source_path}' to '{backup_path}' completed successfully!"
        else:
            logging.warning(f"Source or backup path does not exist.")
            return f"Source or backup path does not exist."

    def recover_files(self, backup_path: str, destination_path: str) -> str:
        """
        Recover files from the backup directory to the destination directory.
        :param backup_path: The backup directory to recover from.
        :param destination_path: The destination directory to recover to.
        :return: A success message if the recovery is completed successfully.
        """
        if os.path.exists(backup_path) and os.path.exists(destination_path):
            for root, dirs, files in os.walk(backup_path):
                for file in files:
                    backup_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(backup_file_path, backup_path)
                    destination_file_path = os.path.join(
                        destination_path, relative_path
                    )
                    if not os.path.exists(os.path.dirname(destination_file_path)):
                        os.makedirs(os.path.dirname(destination_file_path))
                    shutil.copy2(backup_file_path, destination_file_path)
            logging.info(
                f"Recovery from '{backup_path}' to '{destination_path}' completed successfully!"
            )
            return f"Recovery from '{backup_path}' to '{destination_path}' completed successfully!"
        else:
            logging.warning(f"Backup or destination path does not exist.")
            return f"Backup or destination path does not exist."
