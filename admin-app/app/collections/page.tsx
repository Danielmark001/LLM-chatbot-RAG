"use client";

import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { AiOutlineFileAdd } from "react-icons/ai";
import { CiSearch } from "react-icons/ci";
import DocumentPopup from '../components/DocumentPopup';
import FileCard from '../components/FileCard';
import FileDeletionPopup from '../components/FileDeletionPopup';

/**
 * Document interface representing file metadata
 */
interface Document {
  _id: string;
  file: string;
  url: string;
}

/**
 * FilesList component for managing files within a collection
 */
const FilesList: React.FC = () => {
  // State declarations
  const [documents, setDocuments] = useState<Document[]>([]);
  const [showPopup, setShowPopup] = useState<boolean>(false);
  const [fileCreated, setFileCreated] = useState<boolean>(true);
  const [fileDeleted, setFileDeleted] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [showDeletionPopup, setShowDeletionPopup] = useState<boolean>(false);
  const [selectedFileName, setSelectedFileName] = useState<string>('');
  const [selectedCollection, setSelectedCollection] = useState<string>('');
  const [selectedDocId, setSelectedDocId] = useState<string>('');

  // Get the collection name from URL query parameters
  const searchParams = useSearchParams();
  const collectionName = searchParams.get("query") || '';

  // Event handlers
  const handleFileCreated = () => {
    setFileCreated(!fileCreated);
  };

  const handleFileDeleted = () => {
    setFileDeleted(!fileDeleted);
  };

  const handleAddFileClick = (): void => {
    setShowPopup(true);
  };

  const handleClosePopup = (): void => {
    setShowPopup(false);
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  const handleDeletePress = (id: string, collection: string, file: string) => {
    setSelectedFileName(file);
    setSelectedCollection(collection);
    setSelectedDocId(id);
    setShowDeletionPopup(true);
  };

  const handleCloseDeletionPopup = (): void => {
    setShowDeletionPopup(false);
  };

  // Fetch documents whenever collection changes or files are added/deleted
  useEffect(() => {
    if (collectionName) {
      fetchDocuments();
    }
  }, [collectionName, fileCreated, fileDeleted]);

  /**
   * Fetch all documents in the current collection
   */
  const fetchDocuments = () => {
    fetch(`http://localhost:5000/api/collections/${collectionName}`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch documents');
        }
        return response.json();
      })
      .then((documents: Document[]) => {
        setDocuments(documents);
      })
      .catch(error => {
        console.error('Error fetching documents:', error);
      });
  };

  // Filter documents based on search query
  const filteredFiles = documents.filter(document =>
    document.file.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <main className="flex h-screen flex-col p-24 bg-gray-100">
      {/* Header with collection name and add button */}
      <div className="flex flex-row justify-between">
        <div className="font-semibold relative w-10 text-xl">
          {collectionName}
          <div className="absolute bottom-0 left-2 w-full h-1 bg-[#3F50AD]"></div>
        </div>
        <div>
          {documents.length > 0 && (
            <button 
              onClick={handleAddFileClick} 
              className="bg-[#2C3463] text-white py-2 px-4 rounded-lg font-normal transition-transform duration-300 ease-in-out transform hover:scale-105 hover:bg-[#3C456C]"
            >
              Add New File
            </button>
          )}
        </div>
      </div>

      {/* Display files or empty state */}
      {documents.length > 0 ? (
        <>
          {/* Search bar */}
          <div className='flex items-center justify-center mt-4'>
            <input
              type="text"
              value={searchQuery}
              onChange={handleSearchChange}
              placeholder="Search files..."
              className="border border-gray-300 rounded-lg py-2 px-4 mr-2 w-1/3"
            />
            <CiSearch size={35} />
          </div>

          {/* File list */}
          <div className="flex flex-col mt-5 justify-center items-center overflow-scroll">
            {filteredFiles.map((document: Document, index: number) => (
              <FileCard
                key={index}
                fileName={document.file}
                collectionName={collectionName}
                id={document._id}
                onFileDeleted={handleDeletePress}
                url={document.url}
              />
            ))}
          </div>
        </>
      ) : (
        <div className="flex h-screen flex-col items-center justify-center">
          <div className="flex flex-col bg-white w-4/5 border border-dotted border-[#3F50AD] p-4 mx-auto rounded-lg items-center">
            <AiOutlineFileAdd size={50} color="#2C3463" />
            <p className="font-semibold text-lg mt-2">Upload the materials</p>
            <button
              onClick={handleAddFileClick}
              className="bg-[#2C3463] text-white py-2 px-4 rounded-lg font-normal mt-5 w-2/5 transition-transform duration-300 ease-in-out transform hover:scale-105 hover:bg-[#3C456C]"
            >
              Add New Files
            </button>
          </div>
        </div>
      )}

      {/* Modals/Popups */}
      {showPopup && (
        <DocumentPopup 
          onClose={handleClosePopup} 
          onFileCreated={handleFileCreated} 
          collectionName={collectionName} 
        />
      )}
      
      {showDeletionPopup && (
        <FileDeletionPopup 
          fileName={selectedFileName} 
          collectionName={selectedCollection} 
          id={selectedDocId} 
          onFileDeleted={handleFileDeleted} 
          onClose={handleCloseDeletionPopup} 
        />
      )}
    </main>
  );
};

export default FilesList;
