"use client";

import React,{useState} from 'react';

interface FileProps{
    fileName: string,
    collectionName: string,
    id: string,
    onFileDeleted: (id: string, collection: string, file: string) => void;
    url: string
}

const FileCard: React.FC<FileProps> = ({fileName, collectionName, id, onFileDeleted, url}) => {

  return (
    <div className='border border-black rounded-lg shadow-lg px-7 py-3 mr-5 mt-5 flex w-1/2 justify-between'>
          <a href={url} download={fileName} className="text-blue-600 text-center underline">
              {fileName}
          </a>
        <button className='border border-red-400 rounded-lg px-3 text-red-400 transition-transform duration-300 ease-in-out transform hover:scale-105 '  onClick={() => onFileDeleted(id, collectionName, fileName)}>Delete</button>
    </div>
  );
};

export default FileCard;